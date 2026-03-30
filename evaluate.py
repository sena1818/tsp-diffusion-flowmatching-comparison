"""Evaluate a trained TSP diffusion model."""

import argparse
import os
import sys
import time
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPDiffusionModel
from utils.decode import batch_decode, tour_length, is_valid_tour
from utils.tsp_utils import merge_tours


def compute_gap(pred_tour, opt_tour, coords):
    pred_cost = tour_length(pred_tour, coords)
    opt_cost  = tour_length(opt_tour, coords)
    if opt_cost < 1e-10:
        return 0.0
    return (pred_cost - opt_cost) / opt_cost * 100.0


def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    mode = args.mode or ckpt.get('mode') or saved_args.get('mode', 'flow_matching')
    encoder_type = saved_args.get('encoder_type', 'gated_gcn')
    n_layers     = saved_args.get('n_layers', 12)
    hidden_dim   = saved_args.get('hidden_dim', 256)
    T            = saved_args.get('T', 1000)
    diffusion_schedule = saved_args.get('diffusion_schedule', 'linear')
    inference_schedule = saved_args.get('inference_schedule', 'cosine')

    inference_steps = args.inference_steps or (20 if mode == 'flow_matching' else 50)

    print(f'Mode: {mode} | Encoder: {encoder_type} | Epoch: {ckpt.get("epoch", "?")}')
    print(f'Inference steps: {inference_steps} | Schedule: {inference_schedule} | Device: {device}')

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
    print(f'Loaded {state_key} from checkpoint.')

    dataset = TSPDataset(args.data_file)
    loader  = DataLoader(
        dataset, batch_size=1,  # DIFUSCO official test uses batch_size=1
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f'Dataset: {args.data_file} ({len(dataset)} instances)')

    n_samples = getattr(args, 'n_samples', 1)
    if n_samples > 1:
        print(f'Multi-sample mode: {n_samples} samples per instance (take best)')

    all_gaps = []
    all_valid = []
    total_infer_time = 0.0

    for batch_idx, (coords, adj_0, tours_gt) in enumerate(loader):
        coords = coords.to(device)
        B, N, _ = coords.shape

        best_tours_batch = [None] * B
        best_lengths_batch = [float('inf')] * B

        t_start = time.time()
        for sample_idx in range(n_samples):
            with torch.no_grad():
                heatmaps = model.sample(coords, inference_steps=inference_steps,
                                        inference_trick=getattr(args, 'inference_trick', None))

            if args.decode == 'merge':
                pred_tours = []
                for i in range(B):
                    h = heatmaps[i].cpu()
                    c = coords[i].cpu()
                    tour = merge_tours(h, c)
                    if args.use_2opt:
                        from utils.decode import two_opt_improve
                        tour = two_opt_improve(tour, c)
                    pred_tours.append(tour)
            else:
                pred_tours = batch_decode(
                    heatmaps, coords,
                    method=args.decode,
                    beam_k=args.beam_k,
                    use_2opt=args.use_2opt,
                )

            # Keep the best valid tour across samples.
            for i in range(B):
                pred = pred_tours[i]
                c = coords[i].cpu()
                if is_valid_tour(pred, N):
                    length = tour_length(pred, c)
                    if length < best_lengths_batch[i]:
                        best_lengths_batch[i] = length
                        best_tours_batch[i] = pred

        total_infer_time += time.time() - t_start

        for i in range(B):
            c    = coords[i].cpu()
            pred = best_tours_batch[i]
            opt  = tours_gt[i].tolist()

            valid = pred is not None and is_valid_tour(pred, N)
            all_valid.append(valid)
            if valid:
                gap = compute_gap(pred, opt, c)
                all_gaps.append(gap)

        if (batch_idx + 1) % 50 == 0 and all_gaps:
            print(f'  [{batch_idx+1}/{len(dataset)}] avg gap: {sum(all_gaps)/len(all_gaps):.2f}%')

    n_total   = len(dataset)
    n_valid   = sum(all_valid)
    avg_gap   = sum(all_gaps) / len(all_gaps) if all_gaps else float('nan')
    best_gap  = min(all_gaps) if all_gaps else float('nan')
    worst_gap = max(all_gaps) if all_gaps else float('nan')
    avg_ms    = total_infer_time / n_total * 1000

    import statistics
    std_gap = statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0.0

    decode_str = args.decode + ('+2opt' if args.use_2opt else '')
    data_name  = os.path.basename(args.data_file).replace('.txt', '')

    print(f'\n{"="*55}')
    print(f'{data_name} Results ({n_total} instances, decoder={decode_str}):')
    print(f'  Mode               : {mode}')
    print(f'  Encoder            : {encoder_type}')
    print(f'  Avg Optimality Gap : {avg_gap:.2f}% ± {std_gap:.2f}%')
    print(f'  Best Gap           : {best_gap:.2f}%')
    print(f'  Worst Gap          : {worst_gap:.2f}%')
    print(f'  Valid Tour Rate    : {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)')
    print(f'  Avg Inference Time : {avg_ms:.1f} ms/instance')
    print(f'{"="*55}')

    result = {
        'data_file': args.data_file,
        'checkpoint': args.checkpoint,
        'mode': mode,
        'encoder_type': encoder_type,
        'decoder': decode_str,
        'inference_steps': inference_steps,
        'n_samples': n_samples,
        'n_total': n_total,
        'n_valid': n_valid,
        'avg_gap': avg_gap,
        'std_gap': std_gap,
        'best_gap': best_gap,
        'worst_gap': worst_gap,
        'valid_rate': n_valid / n_total,
        'avg_infer_ms': avg_ms,
        'all_gaps': all_gaps,
    }
    if args.save_result:
        os.makedirs(os.path.dirname(args.save_result) or '.', exist_ok=True)
        with open(args.save_result, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Results saved to: {args.save_result}')

    return result


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate TSP Diffusion Model')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_file',  type=str, required=True)
    p.add_argument('--mode',       type=str, default=None,
                   choices=['flow_matching', 'discrete_ddpm', 'continuous_ddpm'])
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--inference_steps', type=int, default=None)
    p.add_argument('--decode',     type=str, default='merge',
                   choices=['greedy', 'beam_search', 'merge'])
    p.add_argument('--beam_k',     type=int, default=5)
    p.add_argument('--use_2opt',   action='store_true')
    p.add_argument('--inference_trick', type=str, default=None,
                   choices=[None, 'ddim'],
                   help='Gaussian inference mode: None=DDPM stochastic (default), ddim=deterministic')
    p.add_argument('--n_samples',   type=int, default=1,
                   help='Number of parallel samples per instance (take best tour)')
    p.add_argument('--save_result', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
