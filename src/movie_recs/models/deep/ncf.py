"""Neural Collaborative Filtering model."""

from __future__ import annotations

import torch
import torch.nn as nn


class NCF(nn.Module):
    """NCF implementation migrated from the DL notebook."""

    def __init__(self, n_users: int, m_items: int, n_factors: int, hidden_dim: int) -> None:
        super().__init__()
        self.m_items = m_items
        self.user_emb_gmf = nn.Embedding(n_users, n_factors)
        self.item_emb_gmf = nn.Embedding(m_items, n_factors)
        self.user_emb_mlp = nn.Embedding(n_users, n_factors)
        self.item_emb_mlp = nn.Embedding(m_items, n_factors)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.out = nn.Linear(n_factors + hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.user_emb_gmf, self.item_emb_gmf, self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, item: torch.Tensor, user: torch.Tensor) -> torch.Tensor:
        u_gmf = self.user_emb_gmf(user)
        i_gmf = self.item_emb_gmf(item)
        gmf = u_gmf * i_gmf
        u_mlp = self.user_emb_mlp(user)
        i_mlp = self.item_emb_mlp(item)
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))
        return self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
