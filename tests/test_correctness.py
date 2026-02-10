"""
NOVA Winograd F(6,3) HIP Kernel — Correctness Tests

Tests cover: output shapes, accuracy vs FP32 direct conv, NaN/Inf safety,
weight caching, from_conv2d, backward pass, model surgery, training, torch.compile.

Usage:
    pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_winograd import (
    NovaWinogradConv2d,
    NovaWinogradConv2dTrainable,
    NovaWinogradConv2dCompilable,
    nova_forward,
    replace_conv2d_with_nova,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

SHAPE_CONFIGS = [
    (1, 64, 64, 56, 56, 1),
    (1, 128, 128, 28, 28, 1),
    (1, 256, 256, 14, 14, 1),
    (1, 512, 512, 7, 7, 1),
    (4, 64, 64, 56, 56, 1),
    (8, 128, 256, 28, 28, 1),
    (1, 3, 64, 224, 224, 1),
    (2, 64, 64, 32, 32, 1),
]


@pytest.mark.parametrize("B,C,K,H,W,pad", SHAPE_CONFIGS)
def test_basic_shapes(B, C, K, H, W, pad):
    """Output shapes are correct and free of NaN/Inf."""
    conv = NovaWinogradConv2d(C, K, kernel_size=3, padding=pad).cuda()
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float16)
    out = conv(x)
    H_out = H + 2 * pad - 2
    W_out = W + 2 * pad - 2
    assert out.shape == (B, K, H_out, W_out)
    assert torch.isnan(out).sum().item() == 0
    assert torch.isinf(out).sum().item() == 0


ACCURACY_CONFIGS = [
    ("conv2_x", 1, 64, 64, 56, 56),
    ("conv3_x", 1, 128, 128, 28, 28),
    ("conv4_x", 1, 256, 256, 14, 14),
    ("conv5_x", 1, 512, 512, 7, 7),
    ("batch8", 8, 64, 64, 56, 56),
]


@pytest.mark.parametrize("name,B,C,K,H,W", ACCURACY_CONFIGS)
def test_accuracy_vs_direct_conv(name, B, C, K, H, W):
    """NOVA output within 6% of FP32 direct conv, zero NaN/Inf."""
    conv_ref = nn.Conv2d(C, K, 3, padding=1, bias=False).cuda()
    x = torch.randn(B, C, H, W, device="cuda")

    with torch.no_grad():
        ref = conv_ref(x)

    nova_out = nova_forward(x.half(), conv_ref.weight.data.float().contiguous(), padding=1)
    rel_err = (torch.norm(nova_out.float() - ref) / torch.norm(ref)).item()

    assert rel_err < 0.06, f"{name}: rel_err={rel_err:.6f}"
    assert torch.isnan(nova_out).sum().item() == 0
    assert torch.isinf(nova_out).sum().item() == 0


NAN_CONFIGS = [
    (1, 64, 64, 56, 56),
    (1, 128, 128, 28, 28),
    (1, 256, 256, 14, 14),
    (1, 512, 512, 7, 7),
    (8, 64, 128, 56, 56),
    (4, 256, 512, 14, 14),
]


@pytest.mark.parametrize("B,C,K,H,W", NAN_CONFIGS)
def test_zero_nan_inf(B, C, K, H, W):
    """Zero NaN/Inf across diverse configurations."""
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float16)
    w = torch.randn(K, C, 3, 3, device="cuda", dtype=torch.float32) * 0.5
    out = nova_forward(x, w, padding=1)
    assert torch.isnan(out).sum().item() == 0
    assert torch.isinf(out).sum().item() == 0


def test_weight_caching():
    """Cached weights produce identical output; weight updates are detected."""
    conv = NovaWinogradConv2d(64, 64, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 64, 28, 28, device="cuda", dtype=torch.float16)

    out1 = conv(x).clone()
    out2 = conv(x).clone()
    assert torch.allclose(out1, out2, atol=0)

    with torch.no_grad():
        conv.weight.add_(0.1)
    out3 = conv(x).clone()
    assert not torch.allclose(out1, out3, atol=1e-3)


def test_from_conv2d():
    """from_conv2d preserves weights and produces close outputs."""
    std_conv = nn.Conv2d(64, 128, 3, padding=1, bias=True).cuda()
    nova_conv = NovaWinogradConv2d.from_conv2d(std_conv).cuda()

    assert torch.allclose(std_conv.weight.data, nova_conv.weight.data)
    assert torch.allclose(std_conv.bias.data, nova_conv.bias.data)

    x = torch.randn(1, 64, 28, 28, device="cuda")
    with torch.no_grad():
        ref = std_conv(x)
        nova_out = nova_conv(x.half())

    rel_err = (torch.norm(nova_out.float() - ref) / torch.norm(ref)).item()
    assert rel_err < 0.06


def test_backward_pass():
    """Trainable module produces valid gradients."""
    C, K = 32, 64
    nova_conv = NovaWinogradConv2dTrainable(C, K, kernel_size=3, padding=1, bias=True).cuda()
    ref_conv = nn.Conv2d(C, K, 3, padding=1, bias=True).cuda()
    ref_conv.weight.data.copy_(nova_conv.weight.data)
    ref_conv.bias.data.copy_(nova_conv.bias.data)

    x = torch.randn(2, C, 14, 14, device="cuda", requires_grad=True)

    ref_out = ref_conv(x)
    ref_out.sum().backward()
    ref_gi = x.grad.clone()
    x.grad = None

    nova_out = nova_conv(x)
    nova_out.float().sum().backward()
    nova_gi = x.grad.clone()

    gi_err = (torch.norm(nova_gi - ref_gi) / torch.norm(ref_gi)).item()
    gw_err = (torch.norm(nova_conv.weight.grad - ref_conv.weight.grad) /
              torch.norm(ref_conv.weight.grad)).item()

    assert gi_err < 0.10, f"grad_input rel_err={gi_err:.6f}"
    assert gw_err < 0.15, f"grad_weight rel_err={gw_err:.6f}"


def test_replace_conv2d():
    """Model surgery replaces exactly the eligible layers."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False),     # eligible
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),      # NOT eligible
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1, bias=True),    # eligible
        nn.ReLU(),
        nn.Conv2d(128, 256, 1),                           # NOT eligible
    ).cuda()

    n_replaced = replace_conv2d_with_nova(model, verbose=False)
    assert n_replaced == 2
    assert isinstance(model[0], NovaWinogradConv2d)
    assert not isinstance(model[2], NovaWinogradConv2d)
    assert isinstance(model[4], NovaWinogradConv2d)


def test_training_step():
    """Full training loop: loss decreases across SGD steps."""
    conv = NovaWinogradConv2dTrainable(32, 64, kernel_size=3, padding=1, bias=True).cuda()
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.01)
    x = torch.randn(2, 32, 14, 14, device="cuda")
    target = torch.randn(2, 64, 14, 14, device="cuda", dtype=torch.float16)

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        out = conv(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss not decreasing: {losses}"
    assert conv.weight.grad is not None


def test_torch_compile():
    """torch.compile with fullgraph=True works end-to-end."""
    conv = NovaWinogradConv2dCompilable(64, 128, kernel_size=3, padding=1).cuda()
    x = torch.randn(2, 64, 28, 28, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        eager_out = conv(x)

    compiled_conv = torch.compile(conv, fullgraph=True)
    with torch.no_grad():
        compiled_out = compiled_conv(x)

    assert torch.allclose(eager_out, compiled_out, atol=0)
    assert torch.isnan(compiled_out).sum().item() == 0
