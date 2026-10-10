"""Exercise the kernels needed for training before downloading a large model."""
from __future__ import annotations

import argparse
import json
import platform


def check_cuda(bits: int = 4, optimizer: str = "adamw_8bit") -> dict:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES. "
                           "The Spark CUDA 13 build requires a CUDA 13-compatible driver (580 or newer).")
    dtype = torch.float32 if bits == 32 else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    # Allocation alone doesn't detect unsupported cuBLAS/SDPA GPU architectures.
    inputs = torch.randn(2, 4, 32, 64, device="cuda", dtype=dtype, requires_grad=True)
    output = functional.scaled_dot_product_attention(inputs, inputs, inputs)
    output.float().square().mean().backward()
    if not torch.isfinite(inputs.grad).all().item():
        raise RuntimeError("CUDA attention backward produced non-finite gradients.")
    if bits in (4, 8) or optimizer == "adamw_8bit":
        import bitsandbytes as bnb
    if bits in (4, 8):
        if bits == 4:
            layer = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=dtype,
                                     quant_type="nf4", compress_statistics=True)
        else:
            layer = bnb.nn.Linear8bitLt(128, 64, bias=False, has_fp16_weights=False)
        layer = layer.to("cuda")
        activations = torch.randn(8, 128, device="cuda", dtype=dtype, requires_grad=True)
        layer(activations).float().square().mean().backward()
        if not torch.isfinite(activations.grad).all().item():
            raise RuntimeError("Quantized linear backward produced non-finite gradients.")
    # Exceed BnB's small-tensor threshold so its 8-bit CUDA optimizer actually runs.
    parameter = torch.nn.Parameter(torch.ones(8192, device="cuda"))
    optim = (bnb.optim.AdamW8bit([parameter], lr=0.01) if optimizer == "adamw_8bit"
             else torch.optim.AdamW([parameter], lr=0.01))
    parameter.square().mean().backward()
    optim.step()
    torch.cuda.synchronize()
    if not torch.isfinite(parameter).all().item() or torch.equal(parameter, torch.ones_like(parameter)):
        raise RuntimeError("CUDA optimizer failed to update its parameters.")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "architecture": platform.machine(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "dtype": str(dtype), "quantization_bits": bits, "optimizer": optimizer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--optimizer", choices=["adamw_8bit", "adamw"], default="adamw_8bit")
    args = parser.parse_args()
    print(json.dumps(check_cuda(args.quantization_bits, args.optimizer)))


if __name__ == "__main__":
    main()
