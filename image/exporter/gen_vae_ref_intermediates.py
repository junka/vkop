#!/usr/bin/env python
"""Generate ORT reference values for selected VAE decoder intermediates.

Adds the chosen intermediate tensors as extra graph outputs, runs ORT on the
same latent the vkop driver feeds, and saves {tensor_name: fp32 array}.
"""
import argparse
import numpy as np
import onnx
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="vae_decoder_512_clean.onnx")
    ap.add_argument("latent", nargs="?", default="vae_latent.raw")
    ap.add_argument("out", nargs="?", default="vae_ref_intermediates.npz")
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        help="substring of the tensor name to expose (default: stage boundaries)",
    )
    args = ap.parse_args()

    m = onnx.load(args.model)
    g = m.graph
    known = [v.name for v in list(g.value_info) + list(g.output)]
    if args.filter:
        picked = [n for n in known if any(f in n for f in args.filter)]
    else:
        # one tensor per stage: conv_in, mid_block in/out, each up_blocks'
        # final Add, the resample outputs and the tail.
        picks = ["/decoder/conv_in/Conv_output_0",
                 "/decoder/mid_block/resnets.0/Add_output_0",
                 "/decoder/mid_block/attentions.0/Add_output_0",
                 "/decoder/mid_block/resnets.1/Add_output_0"]
        for ub in range(5):
            picks.append(f"/decoder/up_blocks.{ub}/upsampler/resample/resample.0/Resize_output_0")
            picks.append(f"/decoder/up_blocks.{ub}/resnets.2/Add_output_0")
            picks.append(f"/decoder/up_blocks.{ub}/Add_output_0")
        picks += ["/decoder/norm_out/Add_output_0",
                  "/decoder/nonlinearity/Mul_output_0",
                  "/decoder/conv_out/Conv_output_0"]
        picked = [p for p in picks if p in known]
        missing = [p for p in picks if p not in known]
        if missing:
            print("[warn] not in value_info:", missing)
    existing = {o.name for o in g.output}
    for name in picked:
        if name in existing:
            continue
        vi = next(v for v in list(g.value_info) + list(g.output) if v.name == name)
        g.output.extend([vi])
    print(f"[info] exposing {len(picked)} intermediates")

    import os
    probe_path = "vae_decoder_512_probe.onnx"
    if not os.path.exists(probe_path):
        onnx.save_model(
            m,
            probe_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=probe_path + ".data",
        )
    sess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [int(d) for d in inp.shape if isinstance(d, int)] or [1, 64, 1, 32, 32]
    latent = np.fromfile(args.latent, dtype=np.float32).reshape(shape)
    outs = sess.run(None, {inp.name: latent})
    names = [o.name for o in sess.get_outputs()]
    ref = {}
    for n, v in zip(names, outs):
        ref[n] = np.asarray(v, dtype=np.float32).ravel()
        ref[n + "@shape"] = np.asarray(v.shape)
    np.savez(args.out, **ref)
    print(f"[done] wrote {args.out} with {len(picked)} intermediates")


if __name__ == "__main__":
    main()
