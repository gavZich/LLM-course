from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads=1):
    # Each head produces k, q, v vectors of size head_dim.
    head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3 * head_dim)


def kqv(x, linear):
    # Project x into one combined tensor and split it into k, q, v.
    kqv_matrix = linear(x)
    k, q, v = torch.chunk(kqv_matrix, 3, dim=2)
    return k, q, v


def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # Scaled dot-product attention scores: QK^T / sqrt(d)
    A = torch.matmul(b, a.transpose(1, 2)) / math.sqrt(D1)
    return A


def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Lower-triangular mask: token i can attend only to tokens <= i.
    mask = torch.tril(torch.ones(max_context_len, max_context_len))
    mask = mask.unsqueeze(0)
    return mask


def self_attention(v, A, mask=None):
    B, N, D = v.size()

    if mask is not None:
        mask = mask[:, :N, :N]
        A = A.masked_fill(mask == 0, float("-inf"))

    A = F.softmax(A, dim=-1)
    sa = torch.matmul(A, v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa


def multi_head_attention_layer(x, kqv_matrices, mask):
    head_outputs = [
        self_attention_layer(x, kqv_matrix, mask)
        for kqv_matrix in kqv_matrices
    ]

    sa = torch.cat(head_outputs, dim=2)

    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.kqv_matrices = nn.ModuleList(
            [create_kqv_matrix(embed_dim, n_heads) for _ in range(n_heads)]
        )

        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)

        self.n_heads = n_heads
        self.embed_dim = embed_dim

        # Final projection: mix the concatenated head outputs.
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Used only for Part 5 analysis.
        # Shape after forward: (B, n_heads, N, N)
        self.last_attention_weights = None

    def _self_attention_layer_with_weights(self, x, kqv_matrix):
        # Internal helper for analysis.
        # It does not change the public assignment API.
        k, q, v = kqv(x, kqv_matrix)
        att = attention_scores(k, q)

        B, N, D = v.size()
        mask = self.mask[:, :N, :N]
        att = att.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(att, dim=-1)
        sa = torch.matmul(attention_weights, v)

        return sa, attention_weights

    def forward(self, x):
        head_outputs = []
        attention_weights_per_head = []

        for kqv_matrix in self.kqv_matrices:
            head_output, head_attention = self._self_attention_layer_with_weights(
                x,
                kqv_matrix,
            )
            head_outputs.append(head_output)
            attention_weights_per_head.append(head_attention)

        sa = torch.cat(head_outputs, dim=2)
        assert sa.size() == x.size()

        # Store attention weights for interpretability.
        # Shape: (B, n_heads, N, N)
        self.last_attention_weights = torch.stack(
            attention_weights_per_head,
            dim=1,
        ).detach().cpu()

        sa = self.output_proj(sa)
        return sa