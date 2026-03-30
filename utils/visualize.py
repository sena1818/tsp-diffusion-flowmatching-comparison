"""Visualization utilities: diffusion GIF, tour comparison, training curves, and ablation charts."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Dict

import torch


def _coords_to_np(coords) -> np.ndarray:
    """Convert coordinates to (N, 2) numpy array."""
    if isinstance(coords, torch.Tensor):
        return coords.detach().cpu().numpy()
    return np.asarray(coords)


def _draw_tour(ax, xy: np.ndarray, tour: List[int], color: str, lw: float = 1.5, alpha: float = 0.8):
    """Draw a closed tour on an axes."""
    N = len(tour)
    segments = []
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        segments.append([xy[i], xy[j]])
    lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], s=25, c='black', zorder=5)


def _tour_cost(tour: List[int], xy: np.ndarray) -> float:
    """Total Euclidean tour length."""
    total = 0.0
    N = len(tour)
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        diff = xy[i] - xy[j]
        total += float(np.linalg.norm(diff))
    return total


def save_diffusion_gif(
    model,
    coords: torch.Tensor,          # (1, N, 2) or (N, 2)
    output_path: str,
    n_frames: int = 20,
    device=None,
    fps: int = 5,
):
    """Visualize a single flow-matching denoising trajectory and save as a GIF."""
    try:
        import imageio
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio")

    if device is None:
        device = next(model.parameters()).device

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.to(device)

    model.eval()
    xy = _coords_to_np(coords[0])       # (N, 2)
    frames = []

    B, N, _ = coords.shape
    assert B == 1, 'save_diffusion_gif only supports a single instance'

    def maybe_record(frame_list, current_heatmap, current_t):
        frame_list.append((current_t, current_heatmap.detach().cpu().numpy()))

    # Record one trajectory, then subsample if needed.
    trajectory = []

    with torch.no_grad():
        if model.mode == 'flow_matching':
            x = torch.randn(B, N, N, device=device)
            total_steps = model.inference_steps
            for step_idx in range(total_steps):
                t_val = 1.0 - step_idx / total_steps
                heatmap = (x * 0.5 + 0.5).clamp(0, 1)
                heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
                maybe_record(trajectory, heatmap[0], t_val)

                t_tensor = torch.full((B,), t_val, device=device)
                v = model.encoder(coords, x, t_tensor).squeeze(1)
                x = x - (1.0 / total_steps) * v

            heatmap = (x * 0.5 + 0.5).clamp(0, 1)
            heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
            maybe_record(trajectory, heatmap[0], 0.0)
        else:
            raise ValueError(
                'save_diffusion_gif only supports Flow Matching. '
                'Use visualize_diffusion.py for D3PM/DDPM GIFs.'
            )

        if len(trajectory) > n_frames:
            indices = np.linspace(0, len(trajectory) - 1, n_frames, dtype=int)
            trajectory = [trajectory[i] for i in indices]

        for t_val, hm in trajectory:

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.suptitle(f'Flow Matching Denoising  (t = {t_val:.2f})', fontsize=13)

            ax0 = axes[0]
            im = ax0.imshow(hm, vmin=0, vmax=1, cmap='RdYlGn', origin='upper')
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            ax0.set_title('Edge Probability Heatmap')
            ax0.set_xlabel('Node j')
            ax0.set_ylabel('Node i')

            ax1 = axes[1]
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_aspect('equal')
            ax1.set_title('High-Prob Edges on City Map')

            N = hm.shape[0]
            thresh = 0.5
            segs, alphas = [], []
            for i in range(N):
                for j in range(i + 1, N):
                    p = hm[i, j]
                    if p > thresh:
                        segs.append([xy[i], xy[j]])
                        alphas.append(p)
            if segs:
                colors = plt.cm.RdYlGn(np.array(alphas))
                lc = LineCollection(segs, colors=colors, linewidths=1.5)
                ax1.add_collection(lc)
            ax1.scatter(xy[:, 0], xy[:, 1], s=30, c='black', zorder=5)

            plt.tight_layout()

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame = buf.reshape(h, w, 4)[..., :3]   # drop alpha -> RGB
            frames.append(frame)
            plt.close(fig)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f'GIF saved -> {output_path}  ({len(frames)} frames, {fps} fps)')


def plot_tour_comparison(
    coords,
    model_tour: List[int],
    opt_tour: List[int],
    save_path: str,
    title: str = 'Tour Comparison',
):
    """
    Side-by-side plot of optimal tour (blue) and model tour (orange)
    with cost and gap annotated.

    Args:
        coords:     (N, 2) city coordinates
        model_tour: model's predicted tour (0-indexed)
        opt_tour:   optimal tour (0-indexed)
        save_path:  output path, e.g. 'report/figs/tour_cmp.png'
        title:      figure title
    """
    xy = _coords_to_np(coords)
    opt_cost   = _tour_cost(opt_tour, xy)
    model_cost = _tour_cost(model_tour, xy)
    gap = (model_cost - opt_cost) / opt_cost * 100.0 if opt_cost > 1e-10 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=13)

    for ax, tour, color, label in [
        (axes[0], opt_tour,   '#2196F3', f'Optimal  cost={opt_cost:.4f}'),
        (axes[1], model_tour, '#FF9800', f'Model    cost={model_cost:.4f}  gap={gap:.2f}%'),
    ]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=10)
        _draw_tour(ax, xy, tour, color=color)
        if len(tour) <= 30:
            for idx, (x, y) in enumerate(xy):
                ax.annotate(str(idx), (x, y), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, color='dimgray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Tour comparison saved -> {save_path}')


def plot_heatmap(
    coords,
    heatmap,
    title: str = 'Edge Probability Heatmap',
    save_path: str = None,
):
    """
    Visualize an N×N edge probability matrix alongside a city map with high-prob edges.

    Args:
        coords:    (N, 2) city coordinates
        heatmap:   (N, N) edge probability matrix in [0, 1]
        title:     figure title
        save_path: save path; shows interactively if None
    """
    xy = _coords_to_np(coords)
    if isinstance(heatmap, torch.Tensor):
        hm = heatmap.detach().cpu().numpy()
    else:
        hm = np.asarray(heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=13)

    ax0 = axes[0]
    im = ax0.imshow(hm, vmin=0, vmax=1, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title('N×N Edge Probability Matrix')
    ax0.set_xlabel('Node j')
    ax0.set_ylabel('Node i')

    ax1 = axes[1]
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title('High-Prob Edges (p > 0.4)')

    N = hm.shape[0]
    segs, probs = [], []
    for i in range(N):
        for j in range(i + 1, N):
            p = hm[i, j]
            if p > 0.4:
                segs.append([xy[i], xy[j]])
                probs.append(p)
    if segs:
        colors = plt.cm.viridis(np.array(probs))
        lc = LineCollection(segs, colors=colors, linewidths=1.5, alpha=0.8)
        ax1.add_collection(lc)
    ax1.scatter(xy[:, 0], xy[:, 1], s=35, c='red', zorder=5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Heatmap saved -> {save_path}')
    else:
        plt.show()


def plot_training_curve(
    history: dict,
    save_path: str,
    title: str = 'Training Curve',
):
    """
    Plot train/val loss and learning rate schedule.

    Args:
        history:   dict with 'train_loss', 'val_loss', 'lr' lists (from history.json)
        save_path: output path
        title:     figure title
    """
    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss', [])
    lr         = history.get('lr', [])
    epochs     = list(range(1, len(train_loss) + 1))

    has_lr = bool(lr)
    n_rows = 2 if has_lr else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    if train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', color='#2196F3', lw=2)
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, label='Val Loss',
                color='#F44336', lw=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss vs Epoch')

    if has_lr:
        ax2 = axes[1]
        ax2.plot(epochs[:len(lr)], lr, color='#4CAF50', lw=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Learning Rate Schedule')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curve saved -> {save_path}')


def plot_ablation_bar(
    results_dict: Dict[str, float],
    save_path: str,
    metric: str = 'avg_gap',
    title: str = 'Ablation Study',
    ylabel: str = 'Optimality Gap (%)',
):
    """
    Bar chart for ablation / comparison experiments.

    Args:
        results_dict: {name: value} dict, where value is a float or a result dict
                      (metric field will be extracted automatically)
        save_path:    output path
        metric:       key to extract when value is a dict (default: 'avg_gap')
        title:        figure title
        ylabel:       y-axis label
    """
    names, values = [], []
    for name, val in results_dict.items():
        names.append(name)
        if isinstance(val, dict):
            values.append(val.get(metric, float('nan')))
        else:
            values.append(float(val))

    palette = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63',
               '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9
            )

    ax.set_xlabel('Method / Configuration')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.25 + 0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Ablation bar chart saved -> {save_path}')


def plot_generalization_curve(
    sizes: List[int],
    gaps_dict: Dict[str, List[float]],
    save_path: str,
    title: str = 'Generalization across TSP Sizes',
):
    """
    Plot optimality gap vs. TSP problem size for multiple methods.

    Args:
        sizes:      x-axis TSP city counts, e.g. [20, 50, 100]
        gaps_dict:  {method_name: [gap_20, gap_50, gap_100, ...]}
        save_path:  output path
        title:      figure title
    """
    colors  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, gaps) in enumerate(gaps_dict.items()):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        x = sizes[:len(gaps)]
        ax.plot(x, gaps, color=c, marker=m, lw=2, markersize=7, label=name)
        for xi, gi in zip(x, gaps):
            ax.annotate(f'{gi:.2f}%', (xi, gi),
                        textcoords='offset points', xytext=(4, 5),
                        fontsize=8, color=c)

    ax.set_xlabel('TSP Problem Size (N cities)', fontsize=11)
    ax.set_ylabel('Optimality Gap (%)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(sizes)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generalization curve saved -> {save_path}')


if __name__ == '__main__':
    import json

    print('=== visualize.py unit test ===')
    os.makedirs('test_figs', exist_ok=True)

    N = 15
    np.random.seed(42)
    coords = np.random.rand(N, 2)
    opt_tour   = list(range(N))
    model_tour = list(range(N))
    model_tour[3], model_tour[7] = model_tour[7], model_tour[3]

    plot_tour_comparison(coords, model_tour, opt_tour,
                         save_path='test_figs/tour_comparison.png',
                         title=f'TSP-{N} Tour Comparison')

    heatmap = np.random.rand(N, N)
    heatmap = (heatmap + heatmap.T) / 2
    plot_heatmap(coords, heatmap,
                 title=f'TSP-{N} Edge Heatmap',
                 save_path='test_figs/heatmap.png')

    history = {
        'train_loss': [0.9 - i * 0.015 + np.random.rand() * 0.02 for i in range(50)],
        'val_loss':   [0.92 - i * 0.014 + np.random.rand() * 0.02 for i in range(50)],
        'lr':         [1e-3 * (0.95 ** i) for i in range(50)],
    }
    plot_training_curve(history, save_path='test_figs/training_curve.png',
                        title='Flow Matching Training Curve')

    ablation = {
        'FM-GatedGCN':   2.31,
        'FM-GAT':        3.05,
        'FM-GCN':        3.87,
        'D3PM-GatedGCN': 2.58,
        'DDPM-GatedGCN': 4.12,
    }
    plot_ablation_bar(ablation, save_path='test_figs/ablation_bar.png',
                      title='Architecture Ablation (TSP-50)')

    sizes = [20, 50, 100]
    gaps_dict = {
        'Flow Matching': [1.2, 2.3, 4.5],
        'Discrete DDPM': [1.5, 2.8, 5.3],
        'Continuous DDPM': [2.1, 3.9, 7.2],
    }
    plot_generalization_curve(sizes, gaps_dict,
                              save_path='test_figs/generalization.png',
                              title='Generalization across TSP Sizes')

    print('\nAll figures saved to test_figs/.')
    print('Note: save_diffusion_gif requires a trained model; skipped here.')
    print('visualize.py OK')
