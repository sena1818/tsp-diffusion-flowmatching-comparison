"""Generate GIFs and static figures for diffusion trajectories on TSP."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPDiffusionModel
from models.diffusion_schedulers import FMInferenceSchedule, InferenceSchedule
from utils.tsp_utils import merge_tours
from utils.decode import two_opt_improve, tour_length

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not installed, GIF generation will be skipped.")
    print("  Install with: pip install imageio[ffmpeg]")


@torch.no_grad()
def sample_with_intermediates(model, coords, steps=None, record_every=1):
    """
    Run inference and record heatmap snapshots at intermediate steps.

    Returns:
        final_heatmap: (N, N) final edge probability heatmap
        intermediates: list of (t_value, heatmap_NxN) snapshots
    """
    steps = steps or model.inference_steps
    device = coords.device
    B, N, _ = coords.shape
    assert B == 1, "visualization only supports batch_size=1"

    intermediates = []

    if model.mode == 'flow_matching':
        x = torch.randn(B, N, N, device=device)
        for idx, (t_val, dt) in enumerate(FMInferenceSchedule(steps)):
            if idx % record_every == 0:
                h = (x[0] * 0.5 + 0.5).clamp(0, 1)
                h = (h + h.T) / 2
                intermediates.append((t_val, h.cpu()))

            t_tensor = torch.full((B,), t_val, device=device)
            v = model.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = (x * 0.5 + 0.5).clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        intermediates.append((0.0, heatmap[0].cpu()))

    elif model.mode == 'discrete_ddpm':
        xt = torch.randn(B, N, N, device=device)
        xt = (xt > 0).long()

        schedule = InferenceSchedule(
            inference_schedule=model.inference_schedule_type,
            T=model.T, inference_T=steps,
        )

        x0_pred_prob = None
        for i in range(steps):
            t1, t2 = schedule(i)

            if i % record_every == 0:
                h = xt[0].float()
                h = (h + h.T) / 2
                intermediates.append((t1 / model.T, h.cpu()))

            xt_input = xt.float() * 2 - 1
            xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))
            t_tensor = torch.tensor([int(t1)], dtype=torch.float, device=device)
            x0_pred = model.encoder(coords.float(), xt_input.float(), t_tensor)
            x0_pred_prob = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(dim=-1)
            xt = model._categorical_posterior(int(t2), int(t1), x0_pred_prob, xt)

        heatmap = x0_pred_prob[..., 1]
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        heatmap = heatmap.clamp(0, 1)
        intermediates.append((0.0, heatmap[0].cpu()))

    elif model.mode == 'continuous_ddpm':
        xt = torch.randn(B, N, N, device=device)

        schedule = InferenceSchedule(
            inference_schedule=model.inference_schedule_type,
            T=model.T, inference_T=steps,
        )
        diffusion = model.scheduler

        for i in range(steps):
            t1, t2 = schedule(i)

            if i % record_every == 0:
                h = (xt[0] * 0.5 + 0.5).clamp(0, 1)
                h = (h + h.T) / 2
                intermediates.append((t1 / model.T, h.cpu()))

            t_tensor = torch.tensor([int(t1)], dtype=torch.float, device=device)
            pred = model.encoder(coords.float(), xt.float(), t_tensor)
            pred = pred.squeeze(1).clamp(-10.0, 10.0)

            xt = model._gaussian_posterior_ddpm_tensor(
                int(t2), int(t1), pred, xt,
                diffusion.alpha_torch, diffusion.alphabar_torch, diffusion.beta_torch,
            )

        heatmap = (xt.detach() * 0.5 + 0.5).clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        intermediates.append((0.0, heatmap[0].cpu()))

    return heatmap[0].cpu(), intermediates


def plot_heatmap_with_tour(ax, coords, heatmap, tour=None, title="", show_edges=True):
    """Draw top-K edges (blue gradient) and optionally a tour (red) on an axes."""
    N = coords.shape[0]
    c = coords.numpy()

    if show_edges and heatmap is not None:
        h = heatmap.numpy()
        max_edges = 3 * N
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append((i, j, h[i, j]))
        edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges[:max_edges]

        if top_edges:
            probs = np.array([e[2] for e in top_edges])
            p_min, p_max = probs.min(), probs.max()
            if p_max > p_min:
                probs_norm = (probs - p_min) / (p_max - p_min)
            else:
                probs_norm = np.ones_like(probs)

            cmap = plt.cm.Blues
            for e, pn in zip(top_edges, probs_norm):
                color = cmap(0.3 + 0.7 * pn)
                lw = 0.3 + 2.0 * pn
                alpha = 0.15 + 0.7 * pn
                ax.plot([c[e[0], 0], c[e[1], 0]], [c[e[0], 1], c[e[1], 1]],
                        color=color, linewidth=lw, alpha=alpha, zorder=1)

    if tour is not None:
        tour_closed = list(tour) + [tour[0]]
        tour_coords = c[tour_closed]
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], color='#e74c3c',
                linewidth=1.8, alpha=0.9, zorder=2)

    ax.scatter(c[:, 0], c[:, 1], c='#333333', s=25, zorder=3,
               edgecolors='white', linewidths=0.5)

    ax.set_facecolor('#f8f8f8')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def make_diffusion_gif(coords, intermediates, pred_tour, tour_gt, out_path, fps=4):
    """
    Generate denoising GIF animation.

    Per-frame: show top-K edges (K = 3*N) sorted by probability.
    Edge width and alpha scale with normalized probability.
    Blue color scheme (light blue = low prob, dark blue = high prob).
    Last frame overlays predicted tour (orange) and optimal tour (red).
    """
    if not HAS_IMAGEIO:
        print(f"  Skipping GIF (imageio not installed): {out_path}")
        return

    c = coords.numpy()
    N = c.shape[0]
    frames = []

    n_inter = len(intermediates)
    if n_inter > 24:
        indices = np.linspace(0, n_inter - 1, 24, dtype=int)
        intermediates = [intermediates[i] for i in indices]

    max_edges = 3 * N

    for frame_idx, (t_val, heatmap) in enumerate(intermediates):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f8f8')
        h = heatmap.numpy()

        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append((i, j, h[i, j]))

        edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges[:max_edges]

        if top_edges:
            probs = np.array([e[2] for e in top_edges])
            p_min, p_max = probs.min(), probs.max()
            if p_max > p_min:
                probs_norm = (probs - p_min) / (p_max - p_min)
            else:
                probs_norm = np.ones_like(probs)

            segments = [[c[e[0]], c[e[1]]] for e in top_edges]
            linewidths = 0.3 + 2.2 * probs_norm
            alphas = 0.15 + 0.7 * probs_norm

            cmap = plt.cm.Blues
            for seg, pw, pa, pn in zip(segments, linewidths, alphas, probs_norm):
                color = cmap(0.3 + 0.7 * pn)
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                        color=color, linewidth=pw, alpha=pa, zorder=1)

        ax.scatter(c[:, 0], c[:, 1], c='#333333', s=20, zorder=3, edgecolors='white', linewidths=0.5)

        is_last = (frame_idx == len(intermediates) - 1)
        if is_last and pred_tour is not None:
            pred_closed = list(pred_tour) + [pred_tour[0]]
            pc = c[pred_closed]
            ax.plot(pc[:, 0], pc[:, 1], color='#ff9800', linewidth=2.2, alpha=0.9, zorder=2)

        if is_last and tour_gt is not None:
            tour_closed = list(tour_gt) + [tour_gt[0]]
            tc = c[tour_closed]
            ax.plot(tc[:, 0], tc[:, 1], color='#e74c3c', linewidth=1.8, alpha=0.9, zorder=3)

        if is_last:
            legend_handles = [
                plt.Line2D([0], [0], color='#5b8fd1', lw=2, label='Top-K heatmap edges'),
            ]
            if pred_tour is not None:
                legend_handles.append(
                    plt.Line2D([0], [0], color='#ff9800', lw=2.2, label='Predicted tour')
                )
            if tour_gt is not None:
                legend_handles.append(
                    plt.Line2D([0], [0], color='#e74c3c', lw=1.8, label='Optimal tour')
                )
            ax.legend(handles=legend_handles, loc='lower left', fontsize=8, framealpha=0.9)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        step_label = f't = {t_val:.3f}' if not is_last else 't = 0.000 (final)'
        ax.set_title(step_label, fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    if frames:
        for _ in range(5):
            frames.append(frames[-1])

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {out_path} ({len(frames)} frames)")


def make_heatmap_evolution_figure(coords, intermediates, mode_name, out_path):
    """Plot 6 heatmap snapshots across the denoising trajectory."""
    n = len(intermediates)
    n_show = min(6, n)
    indices = np.linspace(0, n - 1, n_show, dtype=int)
    selected = [intermediates[i] for i in indices]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3.5))
    if n_show == 1:
        axes = [axes]

    for ax, (t_val, heatmap) in zip(axes, selected):
        plot_heatmap_with_tour(ax, coords, heatmap, title=f't = {t_val:.3f}')

    fig.suptitle(f'{mode_name} — Denoising Process', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Evolution figure saved: {out_path}")


def make_tour_comparison_figure(
    coords_list, tours_model, tours_opt, gaps, mode_names, out_path
):
    """Side-by-side tour comparison: model predictions vs. optimal."""
    n_instances = len(coords_list)
    n_models = len(mode_names)

    fig, axes = plt.subplots(
        n_instances, n_models + 1,
        figsize=(3.5 * (n_models + 1), 3.5 * n_instances),
    )
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_instances):
        c = coords_list[row]
        opt = tours_opt[row]
        opt_len = tour_length(opt, c)

        ax = axes[row, 0]
        plot_heatmap_with_tour(ax, c, None, tour=opt,
                               title=f'Optimal ({opt_len:.3f})')

        for col, mname in enumerate(mode_names):
            ax = axes[row, col + 1]
            pred = tours_model[row][col]
            if pred is not None:
                pred_len = tour_length(pred, c)
                gap = gaps[row][col]
                plot_heatmap_with_tour(
                    ax, c, None, tour=pred,
                    title=f'{mname}\n{pred_len:.3f} (gap={gap:.1f}%)',
                )
            else:
                ax.set_title(f'{mname}\nN/A')
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Tour comparison saved: {out_path}")


def load_model(ckpt_path, device):
    """Load a model from a checkpoint (mirrors evaluate.py logic)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    mode = ckpt.get('mode') or saved_args.get('mode', 'flow_matching')
    encoder_type = saved_args.get('encoder_type', 'gated_gcn')
    n_layers = saved_args.get('n_layers', 12)
    hidden_dim = saved_args.get('hidden_dim', 256)
    T = saved_args.get('T', 1000)
    diffusion_schedule = saved_args.get('diffusion_schedule', 'linear')
    inference_schedule = saved_args.get('inference_schedule', 'cosine')
    inference_steps = 20 if mode == 'flow_matching' else 50

    model = TSPDiffusionModel(
        mode=mode,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        T=T,
        diffusion_schedule=diffusion_schedule,
        inference_schedule=inference_schedule,
        inference_steps=inference_steps,
    ).to(device)

    state_key = 'ema_state' if 'ema_state' in ckpt else 'model_state'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/tsp50_test.txt')
    parser.add_argument('--instance_idx', type=int, default=0)
    parser.add_argument('--n_instances', type=int, default=3,
                        help='number of instances for tour comparison')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['flow_matching', 'discrete_ddpm', 'continuous_ddpm'],
                        help='visualize one mode only (default: all)')
    parser.add_argument('--out_dir', type=str, default='report/figs')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='samples per instance for best-of-N tour comparison')
    parser.add_argument('--gif_steps', type=int, default=None,
                        help='override inference steps for GIF')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    dataset = TSPDataset(args.data_file)
    print(f"Dataset: {args.data_file} ({len(dataset)} instances)")

    model_configs = {
        'flow_matching': {
            'ckpt': 'checkpoints/flow_matching_gated_gcn/best.pt',
            'label': 'Flow Matching',
            'short': 'fm',
        },
        'discrete_ddpm': {
            'ckpt': 'checkpoints/discrete_ddpm_gated_gcn/best.pt',
            'label': 'D3PM',
            'short': 'd3pm',
        },
        'continuous_ddpm': {
            'ckpt': 'checkpoints/continuous_ddpm_gated_gcn/best.pt',
            'label': 'Continuous DDPM',
            'short': 'ddpm',
        },
    }

    if args.mode:
        model_configs = {args.mode: model_configs[args.mode]}

    seed = 42
    idx = args.instance_idx
    coords_tensor, adj_gt, tour_gt = dataset[idx]
    coords_batch = coords_tensor.unsqueeze(0).to(device)
    tour_gt_list = tour_gt.tolist()

    for mode_name, cfg in model_configs.items():
        ckpt_path = cfg['ckpt']
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {mode_name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"  {cfg['label']} — Generating visualizations")
        print(f"{'='*50}")

        model, _ = load_model(ckpt_path, device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        steps = args.gif_steps or model.inference_steps
        record_every = max(1, steps // 30)

        heatmap, intermediates = sample_with_intermediates(
            model, coords_batch, steps=steps, record_every=record_every
        )

        pred_tour = merge_tours(heatmap, coords_tensor)
        pred_tour = two_opt_improve(pred_tour, coords_tensor)

        gif_path = os.path.join(args.out_dir, f'diffusion_{cfg["short"]}.gif')
        make_diffusion_gif(coords_tensor, intermediates, pred_tour, tour_gt_list, gif_path, fps=4)

        evo_path = os.path.join(args.out_dir, f'heatmap_evolution_{cfg["short"]}.png')
        make_heatmap_evolution_figure(
            coords_tensor, intermediates, cfg['label'], evo_path
        )

    print(f"\n{'='*50}")
    print(f"  Tour Comparison")
    print(f"{'='*50}")

    n_inst = min(args.n_instances, len(dataset))
    coords_list = []
    tours_model_all = []
    tours_opt_all = []
    gaps_all = []
    active_modes = []

    for mode_name, cfg in model_configs.items():
        if os.path.exists(cfg['ckpt']):
            active_modes.append((mode_name, cfg))

    n_samples_cmp = args.n_samples
    print(f"Tour comparison: {n_samples_cmp} samples per instance (best-of-N, merge+2opt)")

    for inst_idx in range(n_inst):
        coords_i, adj_gt_i, tour_gt_i = dataset[inst_idx]
        coords_list.append(coords_i)
        tours_opt_all.append(tour_gt_i.tolist())

        row_tours = []
        row_gaps = []
        for mode_name, cfg in active_modes:
            model, _ = load_model(cfg['ckpt'], device)
            cb = coords_i.unsqueeze(0).to(device)

            best_tour = None
            best_len = float('inf')
            for s in range(n_samples_cmp):
                torch.manual_seed(seed + inst_idx * 100 + s)
                heatmap = model.sample(cb)
                h = heatmap[0].cpu()
                pred = merge_tours(h, coords_i)
                pred = two_opt_improve(pred, coords_i)
                length = tour_length(pred, coords_i)
                if length < best_len:
                    best_len = length
                    best_tour = pred

            opt_len = tour_length(tour_gt_i.tolist(), coords_i)
            gap = (best_len - opt_len) / opt_len * 100.0

            row_tours.append(best_tour)
            row_gaps.append(gap)

        tours_model_all.append(row_tours)
        gaps_all.append(row_gaps)

    mode_labels = [cfg['label'] for _, cfg in active_modes]
    tour_cmp_path = os.path.join(args.out_dir, 'tour_comparison.png')
    make_tour_comparison_figure(
        coords_list, tours_model_all, tours_opt_all, gaps_all,
        mode_labels, tour_cmp_path,
    )

    print(f"\nAll visualizations saved to: {args.out_dir}/")


if __name__ == '__main__':
    main()
