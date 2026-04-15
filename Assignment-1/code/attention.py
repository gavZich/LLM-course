from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math



def create_kqv_matrix(input_vector_dim, n_heads = 1):
    # we merge the three matrix/vectors k, q, v to single matrix.
    head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3 * head_dim)

def kqv(x, linear):
    B, N, D = x.size()
    # TODO compute k, q, and v
    # (can do it in 1 or 2 lines.)
    # create the a common layer
    kqv_matrix = linear(x)
    # split it by dimensions
    k, q, v = torch.chunk(kqv_matrix, 3, dim=2)
    return k, q, v


def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # TODO compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # (can do it in 1 or 2 lines.)
    A = torch.matmul(b, a.transpose(1, 2)) / math.sqrt(D1)
    return A


def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant. 

    mask = torch.tril(torch.ones(max_context_len, max_context_len))
    mask = mask.unsqueeze(0) # TODO replace this line with the creation of a causal mask.
    return mask


def self_attention(v, A, mask = None):
    # TODO compute sa (corresponding to y in the assignemnt text).
    # This should take very few lines of code.
    # As usual, the dimensions of v and of sa are (b x n x d).

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
    B, N, D = x.size()
    # TODO implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.
    #
    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    
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
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        return sa
