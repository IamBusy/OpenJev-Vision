import copy

import torch
from transformers import Qwen3Config, Qwen3Model

from openjev.branch_model import score_token_branches


def test_prefix_reuse_matches_recomputation_in_outputs_and_gradients():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    model = Qwen3Model(config).float().eval()
    head = torch.nn.Linear(32, 1, bias=False)
    other = copy.deepcopy(model)
    other_head = copy.deepcopy(head)
    prefix = torch.tensor([[4, 8, 12, 5]])
    suffixes = [[2, 5, 3], [3, 7], [4, 9, 6, 5]]
    shared, usage = score_token_branches(
        model, head, prefix, suffixes, 0, cached=True, branch_batch_size=2
    )
    repeated, _ = score_token_branches(
        other, other_head, prefix, suffixes, 0, cached=False, branch_batch_size=2
    )
    torch.testing.assert_close(shared, repeated, atol=1e-6, rtol=1e-5)
    unpadded = []
    for suffix in suffixes:
        complete = torch.cat((prefix, torch.tensor([suffix])), dim=1)
        h = other(input_ids=complete, use_cache=False).last_hidden_state[:, -1]
        unpadded.append(other_head(h).squeeze(-1))
    torch.testing.assert_close(shared, torch.cat(unpadded), atol=1e-6, rtol=1e-5)
    shared.square().sum().backward()
    repeated.square().sum().backward()
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(head.weight.grad, other_head.weight.grad, atol=1e-5, rtol=1e-4)
    assert usage["prefix_evaluations"] == 1
    # Changing batch composition/order must not change the corresponding score.
    perm = [2, 0, 1]
    reversed_scores, _ = score_token_branches(
        model, head, prefix, [suffixes[i] for i in perm], 0, cached=True, branch_batch_size=3
    )
    torch.testing.assert_close(reversed_scores, shared[perm], atol=1e-6, rtol=1e-5)
