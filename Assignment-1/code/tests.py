import torch
import attention

def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # a three-dim tensor
    b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # a three-dim tensor
    expected_output = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) / (2 ** 0.5) # a three-dim tensor

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)

