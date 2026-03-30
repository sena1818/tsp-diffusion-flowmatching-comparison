# TSP Diffusion: Flow Matching vs DDPM

Comparing three diffusion-based solvers for the **Travelling Salesman Problem (TSP-50)** — Discrete DDPM (D3PM), Continuous DDPM, and Flow Matching — trained from scratch with a GatedGCN encoder on 50K instances.

Reference architecture: [DIFUSCO (NeurIPS 2023)](https://github.com/Edward-Sun/DIFUSCO).

---

## Results

### Single Sample — `merge_tours + 2-opt`, n=1000 test instances

![Main Results](assets/02_main_results_bar.png)

| Method | Optimality Gap ↓ | Inference Time |
|---|---|---|
| **Discrete DDPM (D3PM)** | **1.90%** | 403 ms (50 steps) |
| Flow Matching | 3.45% | 156 ms (20 steps) |
| Continuous DDPM | 6.21% | 334 ms (50 steps) |

### Multi-Sample — best-of-N, `merge_tours + 2-opt`

Taking the best tour across N independent samples dramatically closes the gap. FM with only 5 inference steps and 32 samples achieves **0.26%** — competitive with D3PM at 16×.

![Multi-Sample](assets/10_multi_sample.png)

| Method | Samples | Gap ↓ | Total Time |
|---|---|---|---|
| **D3PM (50 steps)** | **16×** | **0.16%** | 5.3 s |
| FM (5 steps) | 32× | 0.26% | 8.5 s |
| FM (20 steps) | 16× | 0.46% | 5.6 s |
| Continuous DDPM (50 steps) | 16× | 1.36% | 10.8 s |

> **Key finding:** FM's much shorter inference (5 steps vs 50) lets you run 2× more samples in the same wall-clock budget — making it competitive with D3PM despite worse single-sample quality.

### Speed–Quality Trade-off

![Inference Time](assets/03_inference_time.png)

---

## Diffusion Process Visualizations

The GIFs below show the edge-probability heatmap evolving across inference steps (each frame = one denoising step). Brighter edges = higher probability of being in the tour.

### Discrete DDPM (D3PM) — 50 steps

![D3PM diffusion](assets/diffusion_d3pm.gif)

### Flow Matching — 20 steps

![FM diffusion](assets/diffusion_fm.gif)

### Continuous DDPM — 50 steps

![DDPM diffusion](assets/diffusion_ddpm.gif)

### Heatmap Evolution (D3PM)

![D3PM heatmap](assets/heatmap_evolution_d3pm.png)

### Heatmap Evolution (Flow Matching)

![FM heatmap](assets/heatmap_evolution_fm.png)

### Decoded Tour vs Optimal

![Tour comparison](assets/tour_comparison.png)

---

## Architecture

- **Encoder:** GatedGCN (12 layers, hidden_dim=256). Ablation variants: GAT, SimpleGCN.
- **Diffusion modes:**
  - `discrete_ddpm` — D3PM with absorbing-state transitions on {0,1} edge labels
  - `continuous_ddpm` — Gaussian DDPM with stochastic posterior sampling (DDPM, not DDIM)
  - `flow_matching` — ODE-based flow matching on {−1, +1} scaled edge predictions
- **Decoder:** `merge_tours` (DIFUSCO-style Hamiltonian merge) + `2-opt` local search
- **Training:** AdamW lr=2e-4, cosine decay, EMA decay=0.999, 50 epochs, RTX 4090
- **Data:** TSP-50, 50K train / 1K test

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, PyTorch Geometric.

---

## Data

Pre-generated data is in `data/`. To regenerate:

```bash
python data/generate_tsp_data.py --n 50 --num_samples 50000 --split train
python data/generate_tsp_data.py --n 50 --num_samples 1000  --split test
```

---

## Training

```bash
# Discrete DDPM (D3PM)
python train.py --mode discrete_ddpm   --n_nodes 50 --hidden_dim 256 --n_layers 12

# Flow Matching
python train.py --mode flow_matching   --n_nodes 50 --hidden_dim 256 --n_layers 12

# Continuous DDPM
python train.py --mode continuous_ddpm --n_nodes 50 --hidden_dim 256 --n_layers 12
```

Checkpoints are saved to `checkpoints/<run_name>/`. Training history (train/val loss per epoch) is saved as `history.json`.

---

## Evaluation

```bash
# Single sample, merge+2opt
python evaluate.py \
    --mode discrete_ddpm \
    --checkpoint checkpoints/discrete_ddpm_gated_gcn/best.pt \
    --decode_strategy merge2opt \
    --n_samples 1

# Multi-sample (best-of-16)
python evaluate.py \
    --mode discrete_ddpm \
    --checkpoint checkpoints/discrete_ddpm_gated_gcn/best.pt \
    --decode_strategy merge2opt \
    --n_samples 16
```

---

## Visualization

```bash
# Generate diffusion GIF (edge heatmap evolving across steps)
python visualize_diffusion.py \
    --mode flow_matching \
    --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
    --output assets/diffusion_fm.gif

# Heatmap evolution grid
python visualize_diffusion.py \
    --mode discrete_ddpm \
    --plot_type heatmap_evolution

# Tour comparison (our solution vs optimal)
python visualize_diffusion.py \
    --mode discrete_ddpm \
    --plot_type tour_comparison
```

---

## Repository Structure

```
.
├── train.py                      # Training loop (EMA, AdamW, cosine scheduler)
├── evaluate.py                   # Evaluation: gap computation, multi-sample
├── visualize_diffusion.py        # GIF generation, heatmap grids, tour plots
├── models/
│   ├── tsp_model.py              # TSPDiffusionModel — FM / D3PM / DDPM
│   ├── gnn_encoder.py            # GatedGCN / GAT / SimpleGCN encoder
│   ├── diffusion_schedulers.py   # FM, D3PM, Gaussian noise schedules
│   ├── tsp_dataset.py            # Dataset loader (DIFUSCO .txt format)
│   └── nn_utils.py               # GroupNorm32, timestep embedding
├── utils/
│   ├── tsp_utils.py              # TSPEvaluator, merge_tours, gap computation
│   ├── decode.py                 # Greedy, beam search, 2-opt
│   └── visualize.py              # Plotting helpers
├── data/
│   ├── generate_tsp_data.py      # Data generation (elkai / python-tsp)
│   └── tsp{20,50,100}_{train,test}.txt
└── assets/                       # Figures and GIFs used in this README
```
