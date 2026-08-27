"""
Shared fixtures for dispatch tests: matched expert weights across both
container types, a small routed batch, and device selection.
"""
import torch
import pytest

from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, route

T, N_EMBD, HIDDEN, NUM_EXPERTS, TOP_K = 37, 16, 32, 8, 2


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def matched_experts(device):
    """A ModuleListExperts and a StackedExperts(swiglu=False) that compute
    the identical function: GELU(up(x)) @ down, no bias anywhere.
    swiglu=False is required for a fair comparison -- see StackedExperts'
    docstring. Weights are copied directly so this isolates dispatch
    (token routing/grouping) as the only variable between strategies.
    """
    torch.manual_seed(0)
    ml = ModuleListExperts(N_EMBD, HIDDEN, N_EMBD, NUM_EXPERTS, swiglu=False).to(device)
    st = StackedExperts(N_EMBD, HIDDEN, N_EMBD, NUM_EXPERTS, swiglu=False).to(device)
    with torch.no_grad():
        for e in range(NUM_EXPERTS):
            expert = ml[e]
            st.gate_up_proj[e] = expert.f_u.weight.detach().clone()
            st.down_proj[e] = expert.f_d.weight.detach().clone()
    return ml, st


@pytest.fixture(scope="session")
def routed_batch(device):
    """A fixed (x, routing) pair, shared across every strategy under test
    so they're all being compared on identical input.
    """
    torch.manual_seed(0)
    x = torch.randn(T, N_EMBD, device=device)
    gate = GatingNetwork(N_EMBD, NUM_EXPERTS).to(device)
    routing = route(x, gate, TOP_K)
    return x, routing


@pytest.fixture(scope="session")
def dims():
    return {"T": T, "n_embd": N_EMBD, "hidden": HIDDEN, "num_experts": NUM_EXPERTS, "top_k": TOP_K}
