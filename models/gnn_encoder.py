"""GNN encoder for TSP diffusion."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import normalization, zero_module, timestep_embedding


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position encoding for (x,y) coords. Input: (B,N,2) -> (B,N,hidden_dim)."""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, None] / dim_t              # (B, N, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
        return pos  # (B, N, num_pos_feats * 2 = hidden_dim)


class ScalarEmbeddingSine(nn.Module):
    """
    Sinusoidal scalar embedding for edge features (adj_t values) — matches DIFUSCO.

    Input: (B, N, N) -> Output: (B, N, N, num_pos_feats)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t           # (B, N, N, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return pos_x  # (B, N, N, num_pos_feats)


class GatedGCNLayer(nn.Module):
    """Gated GCN layer (Bresson & Laurent, 2017) — matches DIFUSCO GNNLayer, no internal residual."""
    def __init__(self, hidden_dim: int, norm: str = 'layer', learn_norm: bool = True):
        super().__init__()
        d = hidden_dim
        self.A = nn.Linear(d, d, bias=True)
        self.B = nn.Linear(d, d, bias=True)
        self.C = nn.Linear(d, d, bias=True)
        self.U = nn.Linear(d, d, bias=True)
        self.V = nn.Linear(d, d, bias=True)

        if norm == 'layer':
            self.norm_h = nn.LayerNorm(d, elementwise_affine=learn_norm)
            self.norm_e = nn.LayerNorm(d, elementwise_affine=learn_norm)
        elif norm == 'batch':
            self.norm_h = nn.BatchNorm1d(d, affine=learn_norm)
            self.norm_e = nn.BatchNorm1d(d, affine=learn_norm)
        else:
            self.norm_h = None
            self.norm_e = None

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        """h: (B,N,d), e: (B,N,N,d) -> h_out, e_out (no residual)."""
        B, N, d = h.shape

        Uh = self.U(h)   # (B, N, d)

        Vh = self.V(h).unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, d)

        gate_input = (
            self.A(h)[:, :, None, :] +  # source: (B, N, 1, d)
            self.B(h)[:, None, :, :] +  # target: (B, 1, N, d)
            self.C(e)                    # edge:   (B, N, N, d)
        )
        gates = torch.sigmoid(gate_input)   # (B, N, N, d)

        agg = (gates * Vh).sum(dim=2)   # (B, N, d)
        h_out = Uh + agg

        if self.norm_h is not None:
            if isinstance(self.norm_h, nn.BatchNorm1d):
                h_out = self.norm_h(h_out.view(B * N, d)).view(B, N, d)
            else:
                h_out = self.norm_h(h_out)
        if self.norm_e is not None:
            if isinstance(self.norm_e, nn.BatchNorm1d):
                gate_input = self.norm_e(
                    gate_input.view(B * N * N, d)
                ).view(B, N, N, d)
            else:
                gate_input = self.norm_e(gate_input)

        h_out = F.relu(h_out)
        e_out = F.relu(gate_input)

        return h_out, e_out


class GATLayer(nn.Module):
    """Multi-head GAT layer — ablation extension, not in DIFUSCO."""
    def __init__(self, hidden_dim: int, heads: int = 4, **kwargs):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, heads, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        B, N, d = h.shape
        H, Hd = self.heads, self.head_dim
        Q = self.W_q(h).view(B, N, H, Hd).transpose(1, 2)
        K = self.W_k(h).view(B, N, H, Hd).transpose(1, 2)
        V = self.W_v(h).view(B, N, H, Hd).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)
        scores = scores + self.W_e(e).permute(0, 3, 1, 2)
        out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, N, d)
        h_out = F.relu(self.norm(self.out_proj(out)))
        return h_out, e


class SimpleGCNLayer(nn.Module):
    """Lightweight GCN layer — ablation lower-bound baseline, not in DIFUSCO."""
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        w = F.softmax(e.norm(dim=-1, keepdim=True), dim=2)
        agg = (w * h[:, None, :, :]).sum(dim=2)
        h_out = F.relu(self.norm(self.W(h + agg)))
        return h_out, e


class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder aligned with DIFUSCO dense_forward; supports gated_gcn/gat/gcn layer types."""

    def __init__(
        self,
        n_layers: int = 12,
        hidden_dim: int = 256,
        out_channels: int = 1,
        encoder_type: str = 'gated_gcn',
        norm: str = 'layer',
        learn_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.encoder_type = encoder_type
        time_embed_dim = hidden_dim // 2

        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)

        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_embed_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])

        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                nn.SiLU(),
                zero_module(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(n_layers)
        ])

        layer_map = {
            'gated_gcn': lambda: GatedGCNLayer(hidden_dim, norm=norm, learn_norm=learn_norm),
            'gat': lambda: GATLayer(hidden_dim),
            'gcn': lambda: SimpleGCNLayer(hidden_dim),
        }
        if encoder_type not in layer_map:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.layers = nn.ModuleList([layer_map[encoder_type]() for _ in range(n_layers)])

        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
        )

    def forward(
        self,
        coords: torch.Tensor,   # (B, N, 2) city coordinates
        adj_t:  torch.Tensor,   # (B, N, N) diffusion state (preprocessed)
        t:      torch.Tensor,   # (B,) timestep
    ) -> torch.Tensor:
        """coords (B,N,2), adj_t (B,N,N), t (B,) -> (B, out_channels, N, N)."""
        x = self.node_embed(self.pos_embed(coords))       # (B, N, d)
        e = self.edge_embed(self.edge_pos_embed(adj_t))    # (B, N, N, d)

        time_emb = self.time_embed(
            timestep_embedding(t, self.hidden_dim)
        )  # (B, time_embed_dim)

        for layer, time_layer, out_layer in zip(
            self.layers, self.time_embed_layers, self.per_layer_out
        ):
            x_in, e_in = x, e

            x, e = layer(x, e)

            e = e + time_layer(time_emb)[:, None, None, :]

            x = x_in + x
            e = e_in + out_layer(e)

        out = self.out(e.permute(0, 3, 1, 2))
        return out

if __name__ == '__main__':
    B, N = 2, 20
    coords = torch.rand(B, N, 2)
    adj_t = torch.rand(B, N, N)
    t = torch.rand(B) * 1000

    for enc_type in ['gated_gcn', 'gat', 'gcn']:
        for oc in [1, 2]:
            model = GNNEncoder(
                n_layers=4, hidden_dim=128,
                out_channels=oc, encoder_type=enc_type,
            )
            out = model(coords, adj_t, t)
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f'[{enc_type:12s} oc={oc}] '
                f'output: {tuple(out.shape)}  params: {n_params:,}'
            )
    print('GNNEncoder OK')
