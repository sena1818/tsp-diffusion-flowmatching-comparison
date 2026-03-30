"""Schedulers for flow matching, D3PM, and Gaussian DDPM."""

import math
import numpy as np
import torch


class FlowMatchingScheduler:
    """
    Straight-line interpolation scheduler for continuous Flow Matching.

    Forward path: X_t = (1-t)*X_0 + t*epsilon,  t in [0, 1]
    Velocity target: v* = epsilon - X_0  (constant, independent of t)
    Training loss: MSE(v_theta(X_t, t), v*)
    Inference: Euler integration from X_1 ~ N(0,I) to X_0

    Advantages over DDPM/D3PM:
      - Single-direction ODE path, no complex posterior
      - Only ~20 Euler steps needed vs 50 for DDPM/D3PM
      - Constant velocity field is easier to learn
    """

    def interpolate(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation: X_t = (1-t)*x0 + t*epsilon"""
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x0 + t * epsilon

    def get_velocity_target(self, x0: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Constant velocity target: v* = epsilon - x0"""
        return epsilon - x0


class FMInferenceSchedule:
    """
    FM inference timesteps: uniform steps from t=1.0 down to t~0.0.

    Usage:
        for t_val, dt in FMInferenceSchedule(steps=20):
            v = model(coords, x, t_tensor)
            x = x - dt * v
    """
    def __init__(self, inference_steps: int = 20):
        self.steps = inference_steps

    def __iter__(self):
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t_current = 1.0 - i * dt
            yield t_current, dt


class CategoricalDiffusion:
    """
    D3PM discrete diffusion — fully aligned with DIFUSCO CategoricalDiffusion.

    Uses a 2x2 transition matrix Q_bar for binary adjacency matrices:
      Q_t = (1-beta_t)*I + (beta_t/2)*ones   (uniform flip)
      Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t        (cumulative transition)

    Forward: q(x_t | x_0) = x_0_onehot @ Q_bar_t, then Bernoulli sample
    Posterior: q(x_{t-1} | x_t, x_0_hat) via full Bayes formula with Q_bar matrices

    Supports 'linear' and 'cosine' beta schedules.
    """

    def __init__(self, T: int = 1000, schedule: str = 'linear'):
        self.T = T

        if schedule == 'linear':
            b0, bT = 1e-4, 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self._cos_noise(np.arange(0, T + 1, 1)) / self._cos_noise(0)
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        # Cumulative transition matrices.
        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones    # (T, 2, 2) single-step

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)                # (T+1, 2, 2)

    def _cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot: torch.Tensor, t: np.ndarray) -> torch.Tensor:
        """
        Forward noising: q(x_t | x_0). Matches DIFUSCO CategoricalDiffusion.sample().

        Args:
            x0_onehot: (B, N, N, 2) one-hot encoded adjacency matrix
            t:         (B,) numpy int array, range [1, T]
        Returns:
            x_t: (B, N, N) Bernoulli sample in {0, 1}
        """
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
        # Broadcast Q_bar over the dense adjacency tensor.
        xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
        return torch.bernoulli(xt[..., 1].clamp(0, 1))


class InferenceSchedule:
    """
    DDPM/D3PM inference timestep schedule — aligned with DIFUSCO.

    Supports 'linear' and 'cosine' spacing:
      - linear: uniform intervals from T to 1
      - cosine: sine-curved spacing, slower at the start and faster at the end
    """
    def __init__(self, inference_schedule="linear", T=1000, inference_T=50):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)
            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2

        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T
            )
            t1 = np.clip(t1, 1, self.T)
            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T
            )
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2

        else:
            raise ValueError(f"Unknown inference schedule: {self.inference_schedule}")


class GaussianDiffusion:
    """
    Continuous Gaussian diffusion — aligned with DIFUSCO GaussianDiffusion.

    Forward: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Training: epsilon-prediction, MSE loss
    Inference: DDIM deterministic sampling (default) or DDPM stochastic

    Supports 'linear' and 'cosine' beta schedules.

    Precomputed torch tensors are available for all schedule coefficients.
    Call to(device) after model.to(device) to move them to GPU, avoiding
    numpy<->torch conversions in the inference loop.
    """

    def __init__(self, T: int = 1000, schedule: str = 'linear'):
        self.T = T

        if schedule == 'linear':
            b0, bT = 1e-4, 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self._cos_noise(np.arange(0, T + 1, 1)) / self._cos_noise(0)
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
        self.alphabar = np.cumprod(self.alpha)

        # Precompute tensors once to avoid numpy conversions in the loop.
        self.alphabar_torch = torch.from_numpy(self.alphabar).float()
        self.sqrt_alphabar_torch = torch.sqrt(self.alphabar_torch)
        self.sqrt_one_minus_alphabar_torch = torch.sqrt(1.0 - self.alphabar_torch)

        self.alpha_torch = torch.from_numpy(self.alpha).float()
        self.beta_torch = torch.from_numpy(
            np.concatenate([[0.0], self.beta])
        ).float()

    def _cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def to(self, device):
        """Move precomputed inference tensors to the specified device."""
        self.alphabar_torch = self.alphabar_torch.to(device)
        self.sqrt_alphabar_torch = self.sqrt_alphabar_torch.to(device)
        self.sqrt_one_minus_alphabar_torch = self.sqrt_one_minus_alphabar_torch.to(device)
        self.alpha_torch = self.alpha_torch.to(device)
        self.beta_torch = self.beta_torch.to(device)
        return self

    def sample(self, x0: torch.Tensor, t: np.ndarray):
        """
        Forward noising — matches DIFUSCO GaussianDiffusion.sample().

        Args:
            x0: (B, N, N) preprocessed adjacency matrix ({-1,+1} + jitter)
            t:  (B,) numpy int array, range [1, T]
        Returns:
            x_t:     (B, N, N) noisy state
            epsilon: (B, N, N) sampled noise
        """
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
        return xt, epsilon


if __name__ == '__main__':
    B, N = 4, 20

    print('=== FlowMatchingScheduler ===')
    fm = FlowMatchingScheduler()
    adj_0 = torch.zeros(B, N, N)
    t = torch.rand(B)
    eps = torch.randn_like(adj_0)
    x_t = fm.interpolate(adj_0, eps, t)
    v = fm.get_velocity_target(adj_0, eps)
    print(f'  x_t: {tuple(x_t.shape)}, v: {tuple(v.shape)} OK')

    print('=== CategoricalDiffusion ===')
    for sched in ['linear', 'cosine']:
        cd = CategoricalDiffusion(T=1000, schedule=sched)
        import torch.nn.functional as F_
        adj_oh = F_.one_hot(adj_0.long(), num_classes=2).float()
        t_np = np.random.randint(1, 1001, B).astype(int)
        x_t = cd.sample(adj_oh, t_np)
        assert x_t.shape == (B, N, N)
        assert set(x_t.unique().tolist()).issubset({0.0, 1.0})
        print(f'  [{sched:6s}] sample OK, Q_bar shape: {cd.Q_bar.shape}')

    print('=== InferenceSchedule ===')
    for sched in ['linear', 'cosine']:
        iss = InferenceSchedule(inference_schedule=sched, T=1000, inference_T=50)
        pairs = [iss(i) for i in range(50)]
        print(f'  [{sched:6s}] first={pairs[0]}, last={pairs[-1]}')

    print('=== GaussianDiffusion ===')
    for sched in ['linear', 'cosine']:
        gd = GaussianDiffusion(T=1000, schedule=sched)
        adj_scaled = adj_0 * 2 - 1
        t_np = np.random.randint(1, 1001, B).astype(int)
        x_t, eps = gd.sample(adj_scaled, t_np)
        print(f'  [{sched:6s}] sample OK, shape: {tuple(x_t.shape)}')

    print('\nAll schedulers OK')
