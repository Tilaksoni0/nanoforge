"""
Pytest wrapper around check_dispatch_equivalence.py's comparison logic.
Parametrized over DISPATCH_REGISTRY so a new strategy added there is
automatically picked up here with no change to this file.
"""
import pytest
import torch

from benchmarks.check_dispatch_equivalence import _extra_kwargs, STACKED_STRATEGIES
from src.models.moe.dispatch import DISPATCH_REGISTRY, dense_all_experts

TOL = 1e-4


@pytest.fixture(scope="module")
def oracle_output(matched_experts, routed_batch, dims):
    ml, _ = matched_experts
    x, routing = routed_batch
    with torch.no_grad():
        return dense_all_experts(x, routing, ml, dims["top_k"])


@pytest.mark.parametrize("name", [n for n in DISPATCH_REGISTRY if n != "dense_all_experts"])
def test_dispatch_matches_oracle(name, matched_experts, routed_batch, dims, oracle_output):
    ml, st = matched_experts
    x, routing = routed_batch
    fn = DISPATCH_REGISTRY[name]
    experts = st if name in STACKED_STRATEGIES else ml
    kwargs = _extra_kwargs(name, dims["T"])

    with torch.no_grad():
        out = fn(x, routing, experts, dims["top_k"], **kwargs)

    diff = (oracle_output - out).abs().max().item()
    assert diff < TOL, f"{name} disagrees with dense_all_experts oracle (diff={diff:.2e})"
