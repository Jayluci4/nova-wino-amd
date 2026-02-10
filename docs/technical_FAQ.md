# Technical FAQ

### Q1: "Why not just rescale the transform matrices?"

Scaling rows/columns of the transform matrices does not fix the underlying
problem. The condition number is the ratio of the largest to smallest singular
values of the matrix—it's an intrinsic property invariant to row/column scaling.
Standard F(6,3) has condition number ~2,075 for B^T. NOVA reduces it to ~77 by
choosing different interpolation points, which changes the singular value
spectrum itself. This is a 27× improvement that no amount of rescaling can
achieve.

### Q2: "How does this differ from Lavin's original Winograd work?"

Andrew Lavin's paper established the practical framework for Winograd convolution
on GPUs—the tiling scheme, the multi-pass architecture, the GEMM formulation.
That work is foundational and NOVA builds directly on it. The difference is
narrow: NOVA uses the same Cook-Toom algorithm but selects different
interpolation points. Standard points (0, ±1, ±2, ...) minimize integer
complexity but produce ill-conditioned transforms. NOVA searches for real-valued
points that minimize condition numbers. The kernel architecture is identical—only
the ~500 bytes of matrix constants change.

### Q3: "Batch=1 is a niche use case."

Batch=1 is the standard operating point for:
- **Interactive inference** (chatbots, real-time image generation): one request at
  a time, latency is the metric
- **vLLM / continuous batching**: convolution layers in vision encoders process
  individual requests
- **Edge serving**: single-stream deployment on workstation GPUs
- **Stable Diffusion interactive**: users generating one image at a time

The larger point: F(6,3) provides 5.6× fewer arithmetic operations than F(2,3)
at any batch size. At batch=1, NOVA already wins. At larger batches, the gap is a
software optimization problem (kernel fusion, launch overhead)—not a fundamental
limit.

### Q4: "What about F(8,3)?"

Honest answer: F(8,3) remains unsolved for FP16. Standard F(8,3) has condition
number ~196,900. NOVA reduces it 415× to ~474, but this is still too high for
reliable FP16 computation—the norm product is 540 vs. a target of <10.

F(6,3) is the practical sweet spot: 5.6× arithmetic reduction, condition number
77, and verified zero NaN/Inf on 10K ImageNet images. F(8,3) may be viable with
FP32 accumulation throughout, or with future work on mixed-precision transform
strategies.

### Q5: "Why test on SD 1.5? That's an old model."

I also tested on **SDXL** (1024×1024, 38 eligible layers) and
**DenseNet-161** (78 eligible layers, full ImageNetV2 accuracy validation).
Beyond that:
- **DiT models** (SD3, Flux, SORA) use attention, not convolution—Winograd
  doesn't apply to them directly
- **UNet-based models** (SD 1.5, SDXL, ControlNet, inpainting models) are still
  the most deployed diffusion architectures
- The test validates numerical stability through 20+ denoising steps with 49
  convolutions per step—~1,000 F(6,3) transforms per image. Any instability
  would be catastrophically visible.

### Q6: "Can this actually integrate into MIOpen?"

Yes. MIOpen already has the complete multi-pass bidirectional Winograd framework:
- `ConvMPBidirectWinograd<6-3>` and `_xdlops<6-3>` solver classes (C++)
- Assembly transform templates parameterized by tile size
- rocBLAS strided batched GEMM integration
- Performance database infrastructure

None of this code was deleted—it's all still in the repo. It was abandoned
because standard points produce condition number 2,075, making FP16 results
unreliable. The fix is replacing the A, B, G matrix constants (~500 bytes total)
with NOVA values. The architecture, solver framework, and assembly templates are
unchanged.

I even found the experimental FP16 flag:
`MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM` (note the typo in
"EXPEREMENTAL")—confirming FP16 was explored and abandoned.

### Q7: "What about the backward pass?"

Implemented. `NovaWinogradConv2dTrainable` uses HIP forward + native FP32
backward. Gradient errors are <0.03% vs. native PyTorch. Training convergence
verified on CIFAR-10 (>90% accuracy, matching standard training curves).

The backward pass through Winograd transforms is more numerically sensitive than
forward. Using FP32 native backward keeps gradients exact while still
accelerating the forward pass, which dominates inference workloads.

### Q8: "What about depthwise convolutions?"

Depthwise convolutions are memory-bound, not compute-bound—each filter only
operates on one channel, so there's no channel-dimension GEMM to accelerate.
Winograd's advantage is reducing arithmetic intensity of the
channel×channel matrix multiply. For depthwise conv, the arithmetic is already
minimal; the bottleneck is memory bandwidth.

This is why NOVA targets standard (groups=1) 3×3 convolutions: ResNet, DenseNet,
UNet architectures. Models that are primarily depthwise (MobileNet, EfficientNet,
ConvNeXt) don't benefit.

### Q9: "Why HIP instead of assembly?"

Assembly would be the right choice for production integration into MIOpen—and
MIOpen's existing Winograd kernels already use assembly with macro-parameterized
templates. My HIP kernel demonstrates the algorithm and achieves competitive
performance (beats MIOpen F(2,3) at batch=1). Converting to assembly is an
engineering step, not a research step—the numerical constants and tiling strategy
are proven.

### Q10: "torch.compile gives similar speedups. Why write a custom kernel?"

Compiled F(6,3) via `torch.compile` is ~3× slower than MIOpen native at batch=1.
The HIP kernel closes that gap entirely and beats MIOpen. `torch.compile` cannot
fuse across the multi-pass Winograd pipeline (input transform → GEMM → output
transform) the way a dedicated kernel can. It's the difference between generic
compiler optimization and purpose-built GPU code.