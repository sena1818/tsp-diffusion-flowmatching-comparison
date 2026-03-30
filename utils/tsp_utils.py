"""TSP utility functions: tour evaluation, greedy tour construction, and batch gap computation."""

import torch
import numpy as np
from typing import List, Tuple


class TSPEvaluator:

    def __init__(self, coords):
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        self.coords = coords  # (N, 2)
        self.N = coords.shape[0]

    def tour_cost(self, tour: List[int]) -> float:
        """Compute total Euclidean tour length."""
        total = 0.0
        N = len(tour)
        for k in range(N):
            i, j = tour[k], tour[(k + 1) % N]
            diff = self.coords[i] - self.coords[j]
            total += diff.norm().item()
        return total

    def is_valid(self, tour: List[int]) -> bool:
        """Check whether tour is a valid Hamiltonian cycle (visits all cities exactly once)."""
        return len(tour) == self.N and set(tour) == set(range(self.N))

    def optimality_gap(self, pred_tour: List[int], opt_tour: List[int]) -> float:
        """Return (pred_cost - opt_cost) / opt_cost * 100, or 0.0 if opt_cost ~ 0."""
        pred_cost = self.tour_cost(pred_tour)
        opt_cost  = self.tour_cost(opt_tour)
        if opt_cost < 1e-10:
            return 0.0
        return (pred_cost - opt_cost) / opt_cost * 100.0


def merge_tours(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
) -> List[int]:
    """Greedy Hamiltonian cycle construction by globally sorting edges on heatmap/dist score."""
    N = heatmap.shape[0]
    device = heatmap.device

    # Score edges by heatmap / distance.
    c = coords.float()
    diff = c.unsqueeze(1) - c.unsqueeze(0)          # (N, N, 2)
    dist = diff.norm(dim=-1)                          # (N, N)
    dist = dist + torch.eye(N, device=device) * 1e9  # large diagonal to skip self-loops

    score = heatmap / (dist + 1e-8)
    score = score.cpu().numpy()

    rows, cols = np.triu_indices(N, k=1)
    edge_scores = score[rows, cols]
    order = np.argsort(-edge_scores)
    sorted_edges = list(zip(rows[order].tolist(), cols[order].tolist()))

    degree = [0] * N
    adj = [[] for _ in range(N)]

    def find_root(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    parent = list(range(N))

    def union(parent, x, y):
        rx, ry = find_root(parent, x), find_root(parent, y)
        if rx != ry:
            parent[rx] = ry
            return True
        return False

    edges_added = 0
    for u, v in sorted_edges:
        if edges_added == N:
            break
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        if edges_added < N - 1 and find_root(parent, u) == find_root(parent, v):
            continue
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        union(parent, u, v)
        edges_added += 1

    if edges_added < N:
        unvisited = [i for i in range(N) if degree[i] < 2]
        for i in range(0, len(unvisited) - 1, 2):
            u, v = unvisited[i], unvisited[i + 1]
            if degree[u] < 2 and degree[v] < 2:
                adj[u].append(v)
                adj[v].append(u)
                degree[u] += 1
                degree[v] += 1
                edges_added += 1

    tour = [0]
    prev = -1
    current = 0
    for _ in range(N - 1):
        neighbors = [nb for nb in adj[current] if nb != prev]
        if not neighbors:
            remaining = [i for i in range(N) if i not in set(tour)]
            tour.extend(remaining)
            break
        next_node = neighbors[0]
        tour.append(next_node)
        prev = current
        current = next_node

    if len(tour) < N:
        missing = [i for i in range(N) if i not in set(tour)]
        tour.extend(missing)

    return tour[:N]


def compute_batch_gaps(
    pred_tours: List[List[int]],
    opt_tours,          # list[list[int]] or Tensor (B, N)
    coords_batch,       # list[(N,2) tensor] or Tensor (B, N, 2)
) -> Tuple[List[float], List[bool]]:
    """Compute optimality gap and validity for each instance in a batch."""
    B = len(pred_tours)
    gaps   = []
    valids = []

    for i in range(B):
        pred = pred_tours[i]
        opt  = opt_tours[i].tolist() if hasattr(opt_tours[i], 'tolist') else list(opt_tours[i])
        c    = coords_batch[i]
        if hasattr(c, 'cpu'):
            c = c.cpu()

        N = c.shape[0]
        evaluator = TSPEvaluator(c)

        valid = evaluator.is_valid(pred)
        valids.append(valid)

        if valid:
            gap = evaluator.optimality_gap(pred, opt)
            gaps.append(gap)

    return gaps, valids


if __name__ == '__main__':
    torch.manual_seed(42)
    N = 10
    coords = torch.rand(N, 2)

    ev = TSPEvaluator(coords)
    tour = list(range(N))
    print(f'Tour cost:    {ev.tour_cost(tour):.4f}')
    print(f'Tour valid:   {ev.is_valid(tour)}')
    print(f'Gap (self):   {ev.optimality_gap(tour, tour):.2f}%')

    heatmap = torch.rand(N, N)
    heatmap = (heatmap + heatmap.T) / 2
    merged = merge_tours(heatmap, coords)
    print(f'Merge tour:   {merged}')
    print(f'Merge valid:  {ev.is_valid(merged)}')
    print(f'Merge cost:   {ev.tour_cost(merged):.4f}')

    print('tsp_utils.py OK')
