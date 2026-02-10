"""
NOVA Winograd Conv2d modules — drop-in replacements for torch.nn.Conv2d.

Classes:
    NovaWinogradConv2d          — Inference-optimized (workspace caching, weight versioning)
    NovaWinogradConv2dTrainable — Training support (HIP forward, FP32 native backward)
    NovaWinogradConv2dCompilable — torch.compile compatible via custom op
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_winograd.ops import get_lib, _ensure_lib, ptr, compute_tiling


# ─────────────────────────────────────────────────────────────────────────
# torch.compile custom op registration
# ─────────────────────────────────────────────────────────────────────────

_custom_op_registered = False


def _register_custom_op():
    """Register NOVA forward as a torch.library custom op for torch.compile."""
    global _custom_op_registered
    if _custom_op_registered:
        return
    _custom_op_registered = True

    @torch.library.custom_op("nova::winograd_conv2d", mutates_args=())
    def nova_winograd_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        padding: int,
    ) -> torch.Tensor:
        lib = get_lib()
        x = input.half().contiguous() if input.dtype != torch.float16 else input.contiguous()
        w = weight.float().contiguous() if weight.dtype != torch.float32 else weight.contiguous()
        B, C, H, W = x.shape
        K = w.shape[0]
        nh, nw, H_out, W_out = compute_tiling(H, W, padding)
        output = torch.empty(B, K, H_out, W_out, dtype=torch.float16, device=x.device)
        handle = lib.nova_create()
        lib.nova_set_weights(handle, ptr(w), K, C)
        torch.cuda.synchronize()
        lib.nova_forward(handle, ptr(x), ptr(output), B, H, W, padding)
        torch.cuda.synchronize()
        lib.nova_destroy(handle)
        return output

    @nova_winograd_conv2d.register_fake
    def _(input, weight, padding):
        B, C, H, W = input.shape
        K = weight.shape[0]
        H_out = H + 2 * padding - 2
        W_out = W + 2 * padding - 2
        return input.new_empty(B, K, H_out, W_out, dtype=torch.float16)

    def nova_winograd_conv2d_setup_context(ctx, inputs, output):
        input, weight, padding = inputs
        ctx.save_for_backward(input, weight)
        ctx.padding = padding

    def nova_winograd_conv2d_backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        padding = ctx.padding
        go = grad_output.contiguous()
        grad_input = F.conv_transpose2d(go.float(), weight, padding=padding).to(input.dtype)
        inp = input.float().contiguous()
        go32 = go.float().contiguous()
        B, C_in, H, Ww = inp.shape
        K = go32.shape[1]
        grad_weight = torch.zeros_like(weight)
        for b in range(B):
            inp_unf = F.unfold(inp[b:b + 1], 3, padding=padding)
            go_unf = go32[b].reshape(K, -1)
            grad_weight += (go_unf @ inp_unf[0].T).reshape(K, C_in, 3, 3)
        return grad_input, grad_weight, None

    nova_winograd_conv2d.register_autograd(
        nova_winograd_conv2d_backward,
        setup_context=nova_winograd_conv2d_setup_context,
    )


# ─────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────

class NovaWinogradFunction(torch.autograd.Function):
    """Autograd wrapper: HIP forward, FP32 native backward."""

    @staticmethod
    def forward(ctx, input, weight, bias, padding, handle_holder):
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.handle_holder = handle_holder

        x = input.half().contiguous() if input.dtype != torch.float16 else input.contiguous()
        handle_holder._ensure_handle()
        B, C, H, W = x.shape
        nh, nw, H_out, W_out = handle_holder._ensure_workspace(B, H, W)

        lib = get_lib()
        output = torch.empty(
            B, handle_holder.out_channels, H_out, W_out,
            dtype=torch.float16, device=x.device,
        )
        lib.nova_forward_workspace(
            handle_holder._handle,
            ptr(x), ptr(output),
            ptr(handle_holder._V_gemm), ptr(handle_holder._M_gemm),
            B, H, W, padding, nh, nw,
        )
        if bias is not None:
            output = output + bias.half().view(1, -1, 1, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None
        go = grad_output.contiguous()

        if ctx.needs_input_grad[0]:
            grad_input = F.conv_transpose2d(
                go.float(), weight, padding=padding,
            ).to(input.dtype)

        if ctx.needs_input_grad[1]:
            inp = input.float().contiguous()
            go32 = go.float().contiguous()
            B, C_in, H, W = inp.shape
            K = go32.shape[1]
            grad_weight = torch.zeros_like(weight)
            for b in range(B):
                inp_unf = F.unfold(inp[b:b + 1], 3, padding=padding)
                go_unf = go32[b].reshape(K, -1)
                grad_weight += (go_unf @ inp_unf[0].T).reshape(K, C_in, 3, 3)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = go.float().sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


# ─────────────────────────────────────────────────────────────────────────
# NovaWinogradConv2d — Inference module
# ─────────────────────────────────────────────────────────────────────────

class NovaWinogradConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d using NOVA Winograd F(6,3).

    Supports kernel_size=3, stride=1 only.
    Input must be float16. Weights stored in float32, transformed on first use.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Must be 3.
        padding:      Convolution padding (default 1).
        bias:         Whether to include bias (default False).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        assert kernel_size == 3, "Only 3x3 kernels supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self._handle = None
        self._weight_version = -1
        self._V_gemm = None
        self._M_gemm = None
        self._last_config = None

    def _ensure_handle(self):
        """Create handle and update weights if needed."""
        lib = get_lib()
        if self._handle is None:
            self._handle = lib.nova_create()

        current_version = self.weight._version
        if current_version != self._weight_version:
            weight_fp32 = self.weight.data.contiguous()
            lib.nova_set_weights(self._handle, ptr(weight_fp32),
                                 self.out_channels, self.in_channels)
            self._weight_version = current_version
            self._V_gemm = None
            self._M_gemm = None
            self._last_config = None

    def _ensure_workspace(self, batch, H, W):
        """Allocate workspace tensors if config changed."""
        nh, nw, H_out, W_out = compute_tiling(H, W, self.padding)
        P = nh * nw
        BP = batch * P
        config = (batch, H, W, self.padding, nh, nw)

        if self._last_config != config:
            device = self.weight.device
            self._V_gemm = torch.empty(64, self.in_channels, BP,
                                       dtype=torch.float16, device=device)
            self._M_gemm = torch.empty(64, self.out_channels, BP,
                                       dtype=torch.float16, device=device)
            self._last_config = config

        return nh, nw, H_out, W_out

    def forward(self, x):
        assert x.dtype == torch.float16, f"Input must be float16, got {x.dtype}"
        assert x.is_contiguous(), "Input must be contiguous"
        B, C, H, W = x.shape
        assert C == self.in_channels

        self._ensure_handle()
        nh, nw, H_out, W_out = self._ensure_workspace(B, H, W)

        lib = get_lib()
        output = torch.empty(B, self.out_channels, H_out, W_out,
                             dtype=torch.float16, device=x.device)
        lib.nova_forward_workspace(
            self._handle,
            ptr(x), ptr(output),
            ptr(self._V_gemm), ptr(self._M_gemm),
            B, H, W, self.padding, nh, nw,
        )
        if self.bias is not None:
            output = output + self.bias.half().view(1, -1, 1, 1)
        return output

    def __del__(self):
        try:
            if self._handle is not None:
                lib = get_lib()
                lib.nova_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    @classmethod
    def from_conv2d(cls, conv):
        """Create from an existing nn.Conv2d."""
        assert conv.kernel_size == (3, 3), "Only 3x3 kernels supported"
        assert conv.stride == (1, 1), "Only stride=1 supported"
        nova_conv = cls(
            conv.in_channels, conv.out_channels,
            kernel_size=3, padding=conv.padding[0],
            bias=conv.bias is not None,
        )
        nova_conv = nova_conv.to(conv.weight.device)
        nova_conv.weight.data.copy_(conv.weight.data.float())
        if conv.bias is not None:
            nova_conv.bias.data.copy_(conv.bias.data.float())
        return nova_conv

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size=3, padding={self.padding}, tile=F(6,3), points=NOVA"
        )


# ─────────────────────────────────────────────────────────────────────────
# NovaWinogradConv2dTrainable — Training module
# ─────────────────────────────────────────────────────────────────────────

class NovaWinogradConv2dTrainable(NovaWinogradConv2d):
    """NOVA conv with backward pass support for training.

    Forward: HIP kernel (FP16 transforms + FP32 GEMM accumulation).
    Backward: PyTorch native ops in FP32 for gradient stability.
    """

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.float16:
            x = x.half()
        assert x.shape[1] == self.in_channels
        return NovaWinogradFunction.apply(x, self.weight, self.bias, self.padding, self)


# ─────────────────────────────────────────────────────────────────────────
# NovaWinogradConv2dCompilable — torch.compile module
# ─────────────────────────────────────────────────────────────────────────

class NovaWinogradConv2dCompilable(nn.Module):
    """NOVA conv compatible with torch.compile(fullgraph=True).

    Uses torch.library.custom_op so torch.compile can trace through
    the HIP kernel without graph breaks.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        assert kernel_size == 3
        _register_custom_op()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = torch.ops.nova.winograd_conv2d(x, self.weight, self.padding)
        if self.bias is not None:
            out = out + self.bias.half().view(1, -1, 1, 1)
        return out

    @classmethod
    def from_conv2d(cls, conv):
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1)
        nova = cls(
            conv.in_channels, conv.out_channels,
            padding=conv.padding[0], bias=conv.bias is not None,
        )
        nova = nova.to(conv.weight.device)
        nova.weight.data.copy_(conv.weight.data.float())
        if conv.bias is not None:
            nova.bias.data.copy_(conv.bias.data.float())
        return nova

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size=3, padding={self.padding}, tile=F(6,3), points=NOVA, compilable=True"
        )
