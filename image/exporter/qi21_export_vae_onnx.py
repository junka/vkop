"""Export Qwen-Image-2.1 VAE decoder to ONNX.

The VAE decoder converts 64-channel latents (compressed by 8×16×16 in time/space)
into 4-channel output (RGBD/RGBA). At 1024×1024 target resolution, the latent
is 64×8×64×64 (T×H×W after temporal factor 8 and spatial factor 16).

Memory profile from probe_vae.py:
  - 512²: peak 10.7 GB (widest stage 302 MB + DupUp3D transient 151 MB)
  - 1024²: peak 22.9 GB (widest stage 1.2 GB + DupUp3D transient 604 MB)

This exports only the decoder (not the encoder) since image generation only
needs decode. The text encoder is handled separately (Qwen3-VL-7B).

    /Users/doudou/qi21-env/bin/python qi21_export_vae_onnx.py [--size 512|1024]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import qi21_vae as R  # noqa: E402


def load_vae(dtype=torch.float32):
    """Load the VAE decoder from safetensors."""
    cfg_path = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master/vae/config.json")
    ckpt_path = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master/vae/diffusion_pytorch_model.safetensors")

    cfg = json.loads(cfg_path.read_text())
    print(f"[vae] config: z_dim={cfg['z_dim']} base_dim={cfg['base_dim']} "
          f"decoder_base_dim={cfg['decoder_base_dim']}")

    # Build the decoder model
    model = R.AutoencoderKLQwenImage21(**cfg)
    model.eval()

    # Load weights from safetensors
    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[vae] loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
    except ImportError:
        print("[vae] WARNING: safetensors not installed, using random weights")

    model = model.to(dtype=dtype)
    
    # Patch nearest-exact -> nearest for ONNX export compatibility
    _patch_upsample_mode(model)
    
    return model


def _patch_upsample_mode(model):
    """Replace nearest-exact with nearest in all Upsample layers for ONNX export."""
    import torch.nn as nn
    
    for module in model.modules():
        if isinstance(module, nn.Upsample) and hasattr(module, 'mode'):
            if module.mode == 'nearest-exact':
                print(f"[patch] Replacing nearest-exact with nearest in upsample layer")
                module.mode = 'nearest'


def export_decoder(model, size=512, dtype=torch.float32):
    """Export VAE decoder to ONNX with dynamic batch/height/width."""
    # Latent dimensions: B × z_dim × T × H × W
    # For Qwen-Image-2.1: scale_factor_temporal=8, scale_factor_spatial=16
    # Image generation uses single-frame decode (T=1 in latent space)
    scale_t, scale_s = 8, 16
    t_latent = 1  # Single temporal frame for static image
    h_latent = size // scale_s
    w_latent = size // scale_s

    # Create dummy input
    dummy_latent = torch.randn(1, 64, t_latent, h_latent, w_latent, dtype=dtype)

    out_path = HERE / f"vae_decoder_{size}.onnx"
    print(f"[export] latent shape: {tuple(dummy_latent.shape)} -> output: 1×4×{t_latent}×{size}×{size}")

    # Wrap the decode call to avoid tracing through CausalConv3d's squeeze logic
    class DecodeWrapper(torch.nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        
        def forward(self, latent):
            # Call vae.decode which handles all the internal cache logic
            return self.vae.decode(latent, return_dict=False)[0]
    
    wrapper = DecodeWrapper(model)
    wrapper.eval()
    
    # Export with dynamic axes for batch, height, width (time is fixed at 1)
    torch.onnx.export(
        wrapper,
        dummy_latent,
        str(out_path),
        opset_version=18,  # Try higher opset for _upsample_nearest_exact2d support
        input_names=["latent"],
        output_names=["decoded"],
        dynamic_axes={
            "latent": {0: "batch", 3: "height", 4: "width"},
            "decoded": {0: "batch", 3: "height", 4: "width"},
        },
        verbose=False,
        dynamo=False,  # Use legacy tracer (not torch.export) to handle squeeze/unsqueeze
    )

    print(f"[export] saved to {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512, choices=[512, 1024],
                        help="Target resolution (default: 512)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"[vae] loading model in {args.dtype}...")
    model = load_vae(dtype)

    print(f"[vae] exporting decoder at {args.size}×{args.size}...")
    export_decoder(model, size=args.size, dtype=dtype)

    print("[ok] VAE decoder export complete")


if __name__ == "__main__":
    main()
