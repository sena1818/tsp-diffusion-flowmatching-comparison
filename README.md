# TSP Diffusion: Flow Matching vs DDPM Comparison

Comparison of three diffusion-based approaches for the Travelling Salesman Problem (TSP-50):

| Method | Optimality Gap | Inference Steps |
|--------|---------------|-----------------|
| Discrete DDPM (D3PM) | ~1.90% | 50 |
| Flow Matching | ~3.57% | 20 |
| Continuous DDPM | ~6.21% | 50 |

Decoder: `merge_tours + 2-opt` (DIFUSCO-style). Single-sample, 1000 test instances.

## Architecture

- **Encoder**: GatedGCN (12 layers, hidden_dim=256), with GAT and SimpleGCN ablation variants
- **Diffusion**: D3PM (discrete), Gaussian DDPM (continuous), Flow Matching (continuous, {-1,+1} scaling)
- **Training**: AdamW lr=2e-4, cosine decay, EMA decay=0.999, 50 epochs on RTX 4090
- **Data**: TSP-50, 50K training / 1K test instances

Reference implementation: [DIFUSCO (NeurIPS 2023)](https://github.com/Edward-Sun/DIFUSCO)

## Setup

```bash
pip install -r requirements.txt
```

## Data

Pre-generated data is included in `data/`. To regenerate:

```bash
python data/generate_tsp_data.py --n 50 --num_samples 50000 --split train
python data/generate_tsp_data.py --n 50 --num_samples 1000 --split test
```

## Training

```bash
# Flow Matching
python train.py --mode flow_matching --n_nodes 50 --hidden_dim 256 --n_layers 12

# Discrete DDPM (D3PM)
python train.py --mode discrete_ddpm --n_nodes 50 --hidden_dim 256 --n_layers 12

# Continuous DDPM
python train.py --mode continuous_ddpm --n_nodes 50 --hidden_dim 256 --n_layers 12
```

## Evaluation

```bash
python evaluate.py --mode flow_matching --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
    --decode_strategy merge2opt --n_samples 1

python evaluate.py --mode discrete_ddpm --checkpoint checkpoints/discrete_ddpm_gated_gcn/best.pt \
    --decode_strategy merge2opt --n_samples 1
```

## Visualization

```bash
# Generate diffusion GIF
python visualize_diffusion.py --mode flow_matching --checkpoint checkpoints/flow_matching_gated_gcn/best.pt

# Tour comparison plot
python visualize_diffusion.py --mode discrete_ddpm --plot_type tour_comparison
```

## Repository Structure

```
.
├── train.py                  # Training loop (EMA, AdamW, cosine scheduler)
├── evaluate.py               # Evaluation with merge+2opt decoder
├── visualize_diffusion.py    # GIF generation and heatmap visualization
├── models/
│   ├── tsp_model.py          # TSPDiffusionModel (FM / D3PM / DDPM)
│   ├── gnn_encoder.py        # GatedGCN / GAT / SimpleGCN encoder
│   ├── diffusion_schedulers.py  # FM, D3PM, Gaussian schedulers
│   ├── tsp_dataset.py        # Dataset loader (DIFUSCO .txt format)
│   └── nn_utils.py           # GroupNorm32, timestep embedding
├── utils/
│   ├── tsp_utils.py          # TSP evaluator, merge_tours, gap computation
│   ├── decode.py             # Greedy, beam search, 2-opt
│   └── visualize.py          # Plot utilities
└── data/
    ├── generate_tsp_data.py  # Data generation script
    └── tsp{20,50,100}_{train,test}.txt
```
