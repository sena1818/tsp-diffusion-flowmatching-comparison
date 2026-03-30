"""TSP diffusion model — unified entry point for flow matching, discrete DDPM, and continuous DDPM."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import GNNEncoder
from .diffusion_schedulers import (
    FlowMatchingScheduler, FMInferenceSchedule,
    CategoricalDiffusion, InferenceSchedule,
    GaussianDiffusion,
)


class TSPDiffusionModel(nn.Module):
    """Unified model for comparing flow matching, D3PM, and Gaussian DDPM on TSP."""

    MODES = ('flow_matching', 'discrete_ddpm', 'continuous_ddpm')

    def __init__(
        self,
        mode: str = 'flow_matching',
        n_layers: int = 12,
        hidden_dim: int = 256,
        encoder_type: str = 'gated_gcn',
        T: int = 1000,
        diffusion_schedule: str = 'linear',
        inference_schedule: str = 'cosine',
        inference_steps: int = None,
    ):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}"

        self.mode = mode
        self.T = T
        self.diffusion_schedule = diffusion_schedule
        self.inference_schedule_type = inference_schedule
        self.inference_steps = inference_steps or (20 if mode == 'flow_matching' else 50)

        if mode == 'discrete_ddpm':
            out_channels = 2
        else:
            out_channels = 1

        self.encoder = GNNEncoder(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            encoder_type=encoder_type,
        )

        if mode == 'flow_matching':
            self.scheduler = FlowMatchingScheduler()
        elif mode == 'discrete_ddpm':
            self.scheduler = CategoricalDiffusion(T=T, schedule=diffusion_schedule)
        elif mode == 'continuous_ddpm':
            self.scheduler = GaussianDiffusion(T=T, schedule=diffusion_schedule)

    def to(self, *args, **kwargs):
        """Override to() to also move non-nn.Module scheduler tensors to the target device."""
        result = super().to(*args, **kwargs)
        if hasattr(self.scheduler, 'to'):
            device = next(self.parameters()).device
            self.scheduler.to(device)
        return result

    def compute_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        if self.mode == 'flow_matching':
            return self._fm_loss(coords, adj_0)
        elif self.mode == 'discrete_ddpm':
            return self._categorical_loss(coords, adj_0)
        elif self.mode == 'continuous_ddpm':
            return self._gaussian_loss(coords, adj_0)

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        inference_steps: int = None,
        inference_trick: str = None,
    ) -> torch.Tensor:
        """Generate edge probability heatmap. inference_trick: None=DDPM stochastic, 'ddim'=deterministic."""
        steps = inference_steps or self.inference_steps
        if self.mode == 'flow_matching':
            return self._fm_sample(coords, steps)
        elif self.mode == 'discrete_ddpm':
            return self._categorical_sample(coords, steps)
        elif self.mode == 'continuous_ddpm':
            return self._gaussian_sample(coords, steps, inference_trick=inference_trick)

    def _fm_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        B = adj_0.shape[0]
        t = torch.rand(B, device=adj_0.device)
        epsilon = torch.randn_like(adj_0)

        adj_scaled = adj_0 * 2.0 - 1.0

        adj_t = self.scheduler.interpolate(adj_scaled, epsilon, t)

        pred_v = self.encoder(coords, adj_t, t).squeeze(1)

        v_target = self.scheduler.get_velocity_target(adj_scaled, epsilon)
        return F.mse_loss(pred_v, v_target)

    def _fm_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)

        for t_val, dt in FMInferenceSchedule(steps):
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = x * 0.5 + 0.5
        heatmap = heatmap.clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    def _categorical_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        B = adj_0.shape[0]

        t = np.random.randint(1, self.T + 1, B).astype(int)

        adj_matrix_onehot = F.one_hot(adj_0.long(), num_classes=2).float()  # (B,N,N,2)
        xt = self.scheduler.sample(adj_matrix_onehot, t)                     # (B,N,N) {0,1}

        xt_input = xt * 2 - 1
        xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))

        t_tensor = torch.from_numpy(t).float().to(adj_0.device)

        x0_pred = self.encoder(coords.float(), xt_input.float(), t_tensor)

        loss = F.cross_entropy(x0_pred, adj_0.long())
        return loss

    def _categorical_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device

        xt = torch.randn(B, N, N, device=device)
        xt = (xt > 0).long()

        schedule = InferenceSchedule(
            inference_schedule=self.inference_schedule_type,
            T=self.T, inference_T=steps,
        )

        for i in range(steps):
            t1, t2 = schedule(i)
            t1_idx = int(t1)
            t2_idx = int(t2)

            xt_input = xt.float() * 2 - 1
            xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))

            t_tensor = torch.tensor([t1_idx], dtype=torch.float, device=device)
            x0_pred = self.encoder(coords.float(), xt_input.float(), t_tensor)

            x0_pred_prob = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(dim=-1)
            xt = self._categorical_posterior(t2_idx, t1_idx, x0_pred_prob, xt)

        heatmap = x0_pred_prob[..., 1]
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap.clamp(0, 1)

    def _categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """D3PM posterior: q(x_{t-1} | x_t, x_0_hat) via full Bayes formula with Q_bar matrices."""
        diffusion = self.scheduler

        t_idx = int(t)
        tgt_idx = int(target_t) if target_t is not None else t_idx - 1

        device = x0_pred_prob.device

        Q_t = np.linalg.inv(diffusion.Q_bar[tgt_idx]) @ diffusion.Q_bar[t_idx]
        Q_t = torch.from_numpy(Q_t).float().to(device)                    # (2, 2)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t_idx]).float().to(device)   # (2, 2)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[tgt_idx]).float().to(device) # (2, 2)

        xt_onehot = F.one_hot(xt.long(), num_classes=2).float()
        xt_onehot = xt_onehot.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt_onehot, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt_onehot).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt_onehot).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2_new
        ) / x_t_target_prob_part_3_new

        sum_x_t_target_prob = sum_x_t_target_prob + x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if tgt_idx > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        return xt

    def _gaussian_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        B = adj_0.shape[0]

        adj_scaled = adj_0 * 2 - 1
        adj_scaled = adj_scaled * (1.0 + 0.05 * torch.rand_like(adj_scaled))

        t = np.random.randint(1, self.T + 1, B).astype(int)
        xt, epsilon = self.scheduler.sample(adj_scaled, t)

        t_tensor = torch.from_numpy(t).float().to(adj_0.device)
        pred_eps = self.encoder(coords.float(), xt.float(), t_tensor)
        pred_eps = pred_eps.squeeze(1)  # (B, 1, N, N) -> (B, N, N)

        return F.mse_loss(pred_eps, epsilon.float())

    def _gaussian_sample(self, coords: torch.Tensor, steps: int,
                         inference_trick: str = None) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device

        xt = torch.randn(B, N, N, device=device)

        schedule = InferenceSchedule(
            inference_schedule=self.inference_schedule_type,
            T=self.T, inference_T=steps,
        )

        diffusion = self.scheduler
        sqrt_abar = diffusion.sqrt_alphabar_torch          # (T+1,) on device
        sqrt_one_minus_abar = diffusion.sqrt_one_minus_alphabar_torch  # (T+1,)
        abar = diffusion.alphabar_torch                     # (T+1,)

        for i in range(steps):
            t1, t2 = schedule(i)
            t1_idx = int(t1)
            t2_idx = int(t2)

            t_tensor = torch.tensor([t1_idx], dtype=torch.float, device=device)

            with torch.no_grad():
                pred = self.encoder(coords.float(), xt.float(), t_tensor)
                pred = pred.squeeze(1)  # (B, N, N)
                pred = pred.clamp(-10.0, 10.0)

            if inference_trick == 'ddim':
                xt = self._gaussian_posterior_tensor(
                    t2_idx, t1_idx, pred, xt,
                    abar, sqrt_abar, sqrt_one_minus_abar,
                )
            else:
                xt = self._gaussian_posterior_ddpm_tensor(
                    t2_idx, t1_idx, pred, xt,
                    diffusion.alpha_torch, abar, diffusion.beta_torch,
                )

        heatmap = xt.detach() * 0.5 + 0.5
        heatmap = heatmap.clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    @staticmethod
    def _gaussian_posterior_tensor(
        target_t_idx: int, t_idx: int,
        pred: torch.Tensor, xt: torch.Tensor,
        abar: torch.Tensor,
        sqrt_abar: torch.Tensor,
        sqrt_one_minus_abar: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM deterministic update (sigma=0) using precomputed schedule tensors."""
        coeff_xt = (sqrt_abar[target_t_idx] / sqrt_abar[t_idx])
        coeff_eps_sub = sqrt_one_minus_abar[t_idx]
        coeff_eps_add = sqrt_one_minus_abar[target_t_idx]

        xt_target = coeff_xt * (xt - coeff_eps_sub * pred) + coeff_eps_add * pred
        return xt_target

    @staticmethod
    def _gaussian_posterior_ddpm_tensor(
        target_t_idx: int, t_idx: int,
        pred: torch.Tensor, xt: torch.Tensor,
        alpha: torch.Tensor,
        abar: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM stochastic posterior; adds noise z~N(0,I) at all steps except the last."""
        at = alpha[t_idx]
        atbar = abar[t_idx]
        atbar_prev = abar[target_t_idx]
        beta_tilde = beta[t_idx] * (1.0 - atbar_prev) / (1.0 - atbar + 1e-8)

        mean = (1.0 / torch.sqrt(at)) * (
            xt - (1.0 - at) / torch.sqrt(1.0 - atbar + 1e-8) * pred
        )

        if target_t_idx > 0:
            z = torch.randn_like(xt)
            return mean + torch.sqrt(beta_tilde) * z
        return mean

    @torch.no_grad()
    def get_intermediate_heatmap(
        self,
        coords: torch.Tensor,
        target_t: float,
        total_steps: int = 20,
    ) -> torch.Tensor:
        """Return the heatmap at an intermediate FM inference step, for visualization."""
        assert self.mode == 'flow_matching'
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)
        dt = 1.0 / total_steps

        for i in range(total_steps):
            t_val = 1.0 - i * dt
            if t_val < target_t:
                break
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = x * 0.5 + 0.5
        heatmap = heatmap.clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap


if __name__ == '__main__':
    import time as _time

    B, N = 4, 20
    coords = torch.rand(B, N, 2)
    adj_0 = torch.zeros(B, N, N)
    for b in range(B):
        perm = torch.randperm(N)
        for k in range(N):
            i, j = perm[k].item(), perm[(k + 1) % N].item()
            adj_0[b, i, j] = 1.0
            adj_0[b, j, i] = 1.0

    modes = [
        ('flow_matching',   'gated_gcn'),
        ('discrete_ddpm',   'gated_gcn'),
        ('continuous_ddpm', 'gated_gcn'),
        ('flow_matching',   'gat'),
        ('flow_matching',   'gcn'),
    ]

    for mode, enc in modes:
        model = TSPDiffusionModel(
            mode=mode, n_layers=2, hidden_dim=64, encoder_type=enc,
            T=100,
        )
        n_params = sum(p.numel() for p in model.parameters())

        t0 = _time.time()
        loss = model.compute_loss(coords, adj_0)
        loss.backward()
        t_loss = _time.time() - t0

        t0 = _time.time()
        heatmap = model.sample(coords, inference_steps=5)
        t_sample = _time.time() - t0

        assert heatmap.shape == (B, N, N)
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
        sym_err = (heatmap - heatmap.transpose(-1, -2)).abs().max().item()
        assert sym_err < 1e-5

        print(
            f"[{mode:17s} | {enc:10s}] "
            f"params={n_params:,}  loss={loss.item():.4f}  "
            f"t_loss={t_loss*1000:.0f}ms  t_sample={t_sample*1000:.0f}ms"
        )

    print("\nAll TSPDiffusionModel tests passed")
