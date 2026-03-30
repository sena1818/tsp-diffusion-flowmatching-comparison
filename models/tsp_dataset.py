"""TSP dataset reader for DIFUSCO-style text files."""

import torch
from torch.utils.data import Dataset
import numpy as np


class TSPDataset(Dataset):
    def __init__(self, data_file: str):
        """
        Args:
            data_file: path to .txt file, one TSP instance per line
        """
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        # Split coordinates and tour on the marker token.
        coord_part, tour_part = line.split(' output ')

        vals = list(map(float, coord_part.split()))
        N = len(vals) // 2
        coords = np.array(vals, dtype=np.float32).reshape(N, 2)  # (N, 2)

        # Convert to 0-indexed and drop the repeated start node.
        tour_1indexed = list(map(int, tour_part.split()))
        tour = np.array(tour_1indexed[:-1], dtype=np.int64) - 1  # (N,), 0-indexed

        # Mark tour edges in a dense symmetric adjacency matrix.
        adj = np.zeros((N, N), dtype=np.float32)
        for k in range(N):
            i = tour[k]
            j = tour[(k + 1) % N]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        return (
            torch.from_numpy(coords),   # (N, 2)
            torch.from_numpy(adj),      # (N, N)
            torch.from_numpy(tour),     # (N,)
        )


def collate_fn(batch):
    """Standard collate: requires all instances in a batch to have the same N."""
    coords_list, adj_list, tour_list = zip(*batch)
    return (
        torch.stack(coords_list),   # (B, N, 2)
        torch.stack(adj_list),      # (B, N, N)
        torch.stack(tour_list),     # (B, N)
    )
