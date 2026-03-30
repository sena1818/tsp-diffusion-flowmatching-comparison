"""TSP decoders: greedy, beam search, and 2-opt post-processing."""

import math
import torch
import numpy as np
from typing import List


def tour_length(tour: List[int], coords: torch.Tensor) -> float:
    """Total Euclidean length of a tour (0-indexed, length N)."""
    total = 0.0
    N = len(tour)
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        diff = coords[i] - coords[j]
        total += diff.norm().item()
    return total


def is_valid_tour(tour: List[int], N: int) -> bool:
    """Return True if tour is a valid Hamiltonian cycle over N cities."""
    return len(tour) == N and set(tour) == set(range(N))


def greedy_decode(heatmap: torch.Tensor, coords: torch.Tensor) -> List[int]:
    """Greedy decode from node 0, always picking the unvisited neighbor with highest heatmap prob."""
    N = heatmap.shape[0]
    h = heatmap.clone()

    h.fill_diagonal_(0.0)

    visited = [False] * N
    tour = [0]
    visited[0] = True

    for _ in range(N - 1):
        current = tour[-1]
        row = h[current].clone()
        for v in tour:
            row[v] = 0.0

        next_node = row.argmax().item()
        # Fallback for degenerate all-zero rows.
        if visited[next_node]:
            for fallback in range(N):
                if not visited[fallback]:
                    next_node = fallback
                    break
        tour.append(next_node)
        visited[next_node] = True

    return tour


def beam_search_decode(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    k: int = 5,
) -> List[int]:
    """Beam search decode: maintain k candidate paths and return the shortest complete tour."""
    N = heatmap.shape[0]
    h = heatmap.clone()
    h.fill_diagonal_(0.0)

    beams = [(0.0, [0])]

    for step in range(N - 1):
        new_beams = []
        for log_prob, path in beams:
            current = path[-1]
            visited_set = set(path)

            row = h[current].clone()
            for v in visited_set:
                row[v] = 0.0

            remaining = N - len(path)
            top_k = min(k, remaining)
            if top_k == 0:
                continue

            probs, indices = row.topk(top_k)
            for prob, idx in zip(probs.tolist(), indices.tolist()):
                if prob <= 0:
                    continue
                new_log_prob = log_prob + math.log(prob + 1e-10)
                new_beams.append((new_log_prob, path + [idx]))

        if not new_beams:
            # Complete the path if all candidates collapse.
            best_path = max(beams, key=lambda x: x[0])[1]
            remaining_nodes = [v for v in range(N) if v not in set(best_path)]
            beams = [(beams[0][0], best_path + remaining_nodes)]
            break

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:k]

    best_tour = min(
        [b[1] for b in beams if len(b[1]) == N],
        key=lambda t: tour_length(t, coords),
        default=beams[0][1],
    )

    if len(best_tour) < N:
        missing = [v for v in range(N) if v not in set(best_tour)]
        best_tour = best_tour + missing

    return best_tour


def two_opt_improve(tour: List[int], coords: torch.Tensor, max_iter: int = 100) -> List[int]:
    """2-opt local search: repeatedly swap edge pairs until no improvement is found."""
    N = len(tour)
    best = list(tour)
    best_len = tour_length(best, coords)

    for _ in range(max_iter):
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                a, b = best[i], best[(i + 1) % N]
                c, d = best[j], best[(j + 1) % N]

                d_old = (coords[a] - coords[b]).norm() + (coords[c] - coords[d]).norm()
                d_new = (coords[a] - coords[c]).norm() + (coords[b] - coords[d]).norm()

                if d_new < d_old - 1e-10:
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    best_len = best_len - d_old.item() + d_new.item()
                    improved = True

        if not improved:
            break

    return best


def decode_with_2opt(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    method: str = 'greedy',
    beam_k: int = 5,
    max_iter: int = 100,
) -> List[int]:
    """Decode with greedy or beam search, then apply 2-opt improvement."""
    if method == 'greedy':
        tour = greedy_decode(heatmap, coords)
    elif method == 'beam_search':
        tour = beam_search_decode(heatmap, coords, k=beam_k)
    else:
        raise ValueError(f"Unknown method: {method}")

    return two_opt_improve(tour, coords, max_iter=max_iter)


def batch_decode(
    heatmaps: torch.Tensor,     # (B, N, N)
    coords: torch.Tensor,       # (B, N, 2)
    method: str = 'greedy',
    beam_k: int = 5,
    use_2opt: bool = False,
) -> List[List[int]]:
    """Decode a batch of heatmaps and return a list of tours."""
    B = heatmaps.shape[0]
    tours = []
    for i in range(B):
        h = heatmaps[i].cpu()
        c = coords[i].cpu()
        if use_2opt:
            tour = decode_with_2opt(h, c, method=method, beam_k=beam_k)
        elif method == 'beam_search':
            tour = beam_search_decode(h, c, k=beam_k)
        else:
            tour = greedy_decode(h, c)
        tours.append(tour)
    return tours


if __name__ == '__main__':
    import torch

    N = 20
    torch.manual_seed(42)
    coords = torch.rand(N, 2)

    # build a heatmap that peaks along the ground truth tour [0,1,...,N-1]
    tour_gt = list(range(N))
    heatmap = torch.rand(N, N) * 0.1
    for k in range(N):
        i, j = tour_gt[k], tour_gt[(k + 1) % N]
        heatmap[i, j] = 0.9
        heatmap[j, i] = 0.9

    t1 = greedy_decode(heatmap, coords)
    assert is_valid_tour(t1, N), f"Greedy invalid: {t1}"
    print(f'Greedy:      len={tour_length(t1, coords):.4f}  valid={is_valid_tour(t1, N)}')

    t2 = beam_search_decode(heatmap, coords, k=5)
    assert is_valid_tour(t2, N), f"Beam invalid: {t2}"
    print(f'Beam(k=5):   len={tour_length(t2, coords):.4f}  valid={is_valid_tour(t2, N)}')

    t3 = decode_with_2opt(heatmap, coords, method='greedy')
    assert is_valid_tour(t3, N), f"2opt invalid: {t3}"
    print(f'Greedy+2opt: len={tour_length(t3, coords):.4f}  valid={is_valid_tour(t3, N)}')

    print('decode.py OK')
