"""
Low-level ctypes bridge to libnova_winograd.so.

Handles library loading, HIP runtime initialization, and C API bindings.
"""

import ctypes
import os
import torch

# Lazy loading: library loaded on first use to avoid conflicting with torch CUDA init
_lib = None
_LIB_NAME = "libnova_winograd.so"


def _find_lib():
    """Locate the shared library, checking package dir first, then cwd."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", _LIB_NAME),
        os.path.join(os.getcwd(), _LIB_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {_LIB_NAME}. Build it with: make\n"
        f"Searched: {candidates}"
    )


def _ensure_lib():
    """Load shared library lazily after torch has initialized CUDA."""
    global _lib
    if _lib is not None:
        return

    # Force torch to initialize CUDA/HIP runtime first
    torch.cuda.init()

    # Load HIP runtime — prefer system ROCm (matches hipcc version used to build .so),
    # fall back to PyTorch's bundled libs
    rocm_lib_dir = os.environ.get("ROCM_PATH", "/opt/rocm") + "/lib"
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    for dep in ["libhsa-runtime64.so", "libamdhip64.so"]:
        loaded = False
        for lib_dir in [rocm_lib_dir, torch_lib_dir]:
            dep_path = os.path.join(lib_dir, dep)
            if os.path.exists(dep_path):
                try:
                    ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                    break
                except OSError:
                    continue
        if not loaded:
            # Last resort: let the system find it
            try:
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    # Load our library
    lib_path = _find_lib()
    _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    # C API signatures
    _lib.nova_create.restype = ctypes.c_void_p
    _lib.nova_create.argtypes = []
    _lib.nova_destroy.restype = None
    _lib.nova_destroy.argtypes = [ctypes.c_void_p]
    _lib.nova_set_weights.restype = None
    _lib.nova_set_weights.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_forward.restype = None
    _lib.nova_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_forward_workspace.restype = None
    _lib.nova_forward_workspace.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_compute_tiling.restype = None
    _lib.nova_compute_tiling.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    _lib.nova_filter_transform.restype = None
    _lib.nova_filter_transform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_input_transform.restype = None
    _lib.nova_input_transform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_output_transform.restype = None
    _lib.nova_output_transform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.nova_get_K.restype = ctypes.c_int
    _lib.nova_get_K.argtypes = [ctypes.c_void_p]
    _lib.nova_get_C.restype = ctypes.c_int
    _lib.nova_get_C.argtypes = [ctypes.c_void_p]
    _lib.nova_get_U_gemm.restype = ctypes.c_void_p
    _lib.nova_get_U_gemm.argtypes = [ctypes.c_void_p]


def get_lib():
    """Get the loaded ctypes library handle."""
    _ensure_lib()
    return _lib


def ptr(tensor):
    """Get raw GPU pointer from a torch tensor."""
    return ctypes.c_void_p(tensor.data_ptr())


def compute_tiling(H, W, pad):
    """Compute tiling parameters for given spatial size and padding.

    Returns:
        (nh, nw, H_out, W_out) — tile counts and output spatial dimensions.
    """
    _ensure_lib()
    nh = ctypes.c_int()
    nw = ctypes.c_int()
    H_out = ctypes.c_int()
    W_out = ctypes.c_int()
    _lib.nova_compute_tiling(
        H, W, pad,
        ctypes.byref(nh), ctypes.byref(nw),
        ctypes.byref(H_out), ctypes.byref(W_out),
    )
    return nh.value, nw.value, H_out.value, W_out.value


def nova_forward(input_fp16, weight_fp32, padding=1):
    """Functional API: NOVA Winograd F(6,3) forward pass.

    Args:
        input_fp16:  [B, C, H, W] float16 tensor on GPU.
        weight_fp32: [K, C, 3, 3] float32 tensor on GPU.
        padding:     Convolution padding (default 1).

    Returns:
        output: [B, K, H_out, W_out] float16 tensor on GPU.
    """
    _ensure_lib()
    assert input_fp16.is_cuda and input_fp16.dtype == torch.float16
    assert weight_fp32.is_cuda and weight_fp32.dtype == torch.float32
    assert input_fp16.is_contiguous() and weight_fp32.is_contiguous()

    B, C, H, W = input_fp16.shape
    K = weight_fp32.shape[0]
    nh, nw, H_out, W_out = compute_tiling(H, W, padding)

    output = torch.empty(B, K, H_out, W_out, dtype=torch.float16, device=input_fp16.device)

    handle = _lib.nova_create()
    _lib.nova_set_weights(handle, ptr(weight_fp32), K, C)
    torch.cuda.synchronize()
    _lib.nova_forward(handle, ptr(input_fp16), ptr(output), B, H, W, padding)
    torch.cuda.synchronize()
    _lib.nova_destroy(handle)

    return output
