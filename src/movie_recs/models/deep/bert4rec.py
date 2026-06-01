"""BERT4Rec components migrated from the DL notebook."""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class BERT4RecDataset(Dataset):
    """Dataset over per-user interaction sequences."""

    def __init__(self, ds: pd.DataFrame, item2idx: dict[int, int], max_len: int = 100, phase: str = "train") -> None:
        self.ds = ds.reset_index(drop=True)
        self.item2idx = item2idx
        self.max_len = max_len
        self.phase = phase

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds.iloc[idx]
        seq = [self.item2idx[x[0]] for x in row["train_interactions"] if x[0] in self.item2idx][-self.max_len :]
        if self.phase == "train":
            return seq
        targets = {self.item2idx[x[0]] for x in row["test_interactions"] if x[0] in self.item2idx}
        return seq, targets


def collate_fn_train(batch):
    """Pad training sequences."""
    seq_lens = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
    seq_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in batch], batch_first=True, padding_value=0)
    return {"seq_i": seq_padded, "seq_len": seq_lens}


def collate_fn_test(batch):
    """Pad test sequences and keep ground truth sets on CPU."""
    seqs, targets = zip(*batch)
    seq_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    seq_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in seqs], batch_first=True, padding_value=0)
    return {"seq_i": seq_padded, "seq_len": seq_lens, "targets": targets}


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, hidden_dropout_prob: float, hidden_act: str, layer_norm_eps: float) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        self.act = act_map.get(hidden_act, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.dense_1(x))
        out = self.dropout(self.dense_2(out))
        return self.layer_norm(out + x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, hidden_dropout_prob: float, attn_dropout_prob: float, layer_norm_eps: float) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor, return_explanations: bool = False):
        mixed_query = self.query(input_tensor)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)
        query = self.transpose_for_scores(mixed_query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(mixed_key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(mixed_value).permute(0, 2, 1, 3)
        scores = torch.matmul(query, key) / self.sqrt_attention_head_size
        scores = scores + attention_mask
        probs = self.attn_dropout(self.softmax(scores))
        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        hidden_states = self.dense(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return (hidden_states, probs) if return_explanations else hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float, attn_dropout_prob: float, hidden_act: str, layer_norm_eps: float) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, return_explanations: bool = False):
        if return_explanations:
            attention_output, expl = self.multi_head_attention(hidden_states, attention_mask, return_explanations=True)
        else:
            attention_output = self.multi_head_attention(hidden_states, attention_mask)
        output = self.feed_forward(attention_output)
        return (output, expl) if return_explanations else output


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int = 2, n_heads: int = 2, hidden_size: int = 64, inner_size: int = 256, hidden_dropout_prob: float = 0.5, attn_dropout_prob: float = 0.5, hidden_act: str = "gelu", layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_all_encoded_layers: bool = True, return_explanations: bool = False):
        all_layers = []
        for layer_module in self.layer:
            if return_explanations:
                hidden_states, expl = layer_module(hidden_states, attention_mask, return_explanations=True)
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return (all_layers, expl) if return_explanations else all_layers


class BERT4Rec(nn.Module):
    """Notebook-compatible BERT4Rec implementation."""

    def __init__(self, n_items: int, max_seq_length: int = 100) -> None:
        super().__init__()
        self.n_layers = 2
        self.n_heads = 2
        self.hidden_size = 64
        self.inner_size = 128
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = 0.2
        self.hidden_act = "sigmoid"
        self.layer_norm_eps = 1e-5
        self.ITEM_SEQ = "seq_i"
        self.ITEM_SEQ_LEN = "seq_len"
        self.max_seq_length = max_seq_length
        self.mask_ratio = 0.2
        self.loss_type = "CE"
        self.initializer_range = 1e-2
        self.n_items = n_items
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.apply(self._init_weights)

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=gather_index).squeeze(1)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        attention_mask = (item_seq > 0).long()
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        extended = extended.to(dtype=next(self.parameters()).dtype)
        return (1.0 - extended) * -10000.0

    def _neg_sample(self, item_set) -> int:
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length: int):
        return ([0] * (max_length - len(sequence)) + sequence)[-max_length:]

    def reconstruct_train_data(self, item_seq: torch.Tensor):
        device = item_seq.device
        batch_size = item_seq.size(0)
        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_sequences = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                if item == 0:
                    break
                if random.random() < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)
            masked_sequences.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))
        return (
            torch.tensor(masked_sequences, dtype=torch.long, device=device).view(batch_size, -1),
            torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1),
            torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1),
            torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1),
        )

    def reconstruct_test_data(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq

    def forward(self, item_seq: torch.Tensor, return_explanations: bool = False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = self.dropout(self.layer_norm(item_emb + position_embedding))
        attention_mask = self.get_attention_mask(item_seq)
        if return_explanations:
            trm_output, explanations = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True, return_explanations=True)
        else:
            trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return (output, explanations) if return_explanations else output

    def multi_hot_embed(self, masked_index: torch.Tensor, max_length: int) -> torch.Tensor:
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ].long()
        masked_item_seq, pos_items, neg_items, masked_index = self.reconstruct_train_data(item_seq)
        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)
        seq_output = torch.bmm(pred_index_map, seq_output)
        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            targets = (masked_index > 0).float()
            return -torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) / torch.sum(targets)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        test_item_emb = self.item_embedding.weight[: self.n_items]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        targets = (masked_index > 0).float().view(-1)
        return torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) / torch.sum(targets)

    def full_sort_predict(self, interaction, return_explanations: bool = False):
        item_seq = interaction[self.ITEM_SEQ].long()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].long()
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        if return_explanations:
            seq_output, expl = self.forward(item_seq, return_explanations=True)
        else:
            seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[: self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        idxs = item_seq.nonzero()
        item_seq[item_seq == self.n_items] = 0
        scores[idxs[:, 0], item_seq[idxs[:, 0], idxs[:, 1]].long()] = -1000
        return (scores, expl) if return_explanations else scores
