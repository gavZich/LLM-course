from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    # TODO implement this.
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:, :-1] # all the tokens except the last
    labels = batch[:, 1:] # all the tokens except the first
    return (inputs, labels)

def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_bales size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
    # Move vocab dimension to dim=1 as expected by cross_entropy: (B, V, N)
    logits_for_ce = logits.permute(0, 2, 1) # correct to tensor layout expected by cross_entropy
    return F.cross_entropy(logits_for_ce, gold_labels, ignore_index=0)

