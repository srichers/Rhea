#!/usr/bin/env python3
'''Re-export a trained checkpoint's weights under the current source.

A checkpoint saved by save_model is TorchScript, so box3d and predict_all are frozen
into it as they were at training time. Fixing either one leaves every existing
checkpoint on the old behavior. This rebuilds the network from the current source,
loads the old weights into it, and saves it again - no retraining.

The architecture is read back off the checkpoint rather than supplied, so this keeps
working as the config space changes. The two things that cannot be read back are the
activations and the dropout probabilities, because neither has parameters: dropout is
identity at eval and so cannot matter here, while a wrong activation would load
silently. That is what the verification below is for - it compares the rebuilt model
against the original and refuses to save unless they agree exactly.

Usage:
    python3 reexport.py old_cuda.pt newname          # writes newname_{cpu,cuda}.pt
    python3 reexport.py old_cuda.pt newname --devices cpu
    python3 reexport.py old_cuda.pt newname --scalar-activation relu
'''
import argparse

import torch
import torch.nn as nn
import e3nn
import e3nn.o3

from ml_neuralnet import NeuralNetwork
from ml_tools import save_model

# the activations parms may name. Anything not here can be passed by dotted path.
ACTIVATIONS = {
    "silu":    nn.functional.silu,
    "relu":    nn.functional.relu,
    "gelu":    nn.functional.gelu,
    "tanh":    torch.tanh,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}

def resolve_activation(name):
    if name in ACTIVATIONS:
        return ACTIVATIONS[name]
    module, _, attr = name.rpartition(".")
    if not module:
        raise SystemExit(f"unknown activation {name!r}; known: {sorted(ACTIVATIONS)}")
    import importlib
    return getattr(importlib.import_module(module), attr)


def infer_parms(model, scalar_activation, nonscalar_activation):
    '''Recover the constructor parms from a scripted checkpoint.

    Layer counts and irreps come from the built modules themselves, so a checkpoint
    trained with a different width or depth reconstructs without being told.
    '''
    state = model.state_dict()
    parms = {
        "NF":                             int(model.NF),
        "average_heavies_in_final_state":  bool(model.average_heavies_in_final_state),
        "conserve_lepton_number":          bool(model.conserve_lepton_number),
        "random_seed":                     0,   # init is overwritten by the old weights
        "scalar_activation":               scalar_activation,
        "nonscalar_activation":            nonscalar_activation,
    }

    # every shared block has the trunk's width, so the last one's output is irreps_shared
    shared = list(model.TP_activation_stack_shared.named_children())
    parms["nhidden_shared"] = len(shared)
    parms["irreps_shared"]  = e3nn.o3.Irreps(shared[-1][1].irreps_out)

    for stack in ["growthrate", "F4"]:
        blocks = list(getattr(model, f"TP_activation_stack_{stack}").named_children())
        parms[f"nhidden_{stack}"] = len(blocks)
        # build_task uses irreps_<task> only for the layers before the last, so a
        # single-layer head has no intermediate width to recover. Its final block's
        # output is the task's output irreps, not irreps_<task>, and would fail the
        # scalars-first assert, so substitute the trunk's width - it goes unused.
        parms[f"irreps_{stack}"] = (e3nn.o3.Irreps(blocks[0][1].irreps_out)
                                    if len(blocks) > 1 else parms["irreps_shared"])

    # Dropout has no parameters and is identity under eval, so its probability cannot
    # be recovered and cannot affect what this script produces.
    for stack in ["shared", "growthrate", "F4"]:
        parms[f"dropout_{stack}"] = 0.0

    # PETP_Quadratic builds tp_self/tp_flavor/...; PETP_Linear does not
    parms["tensor_product_class"] = (
        "quadratic" if any("tp_self" in k for k in state) else "linear")

    return parms


def verify(old, new, nbatch=64):
    '''Compare the rebuilt network against the original.

    forward() is the network alone, so it isolates the weights from box3d and must
    agree bit for bit - this is what catches a wrong activation, which loads cleanly
    because activations carry no parameters. predict_all is allowed to differ, since
    changing box3d is the whole point, but only by turning a nan into a real number.
    '''
    torch.manual_seed(0)
    x = torch.randn(nbatch, 2, int(old.NF), 10)
    with torch.inference_mode():
        (a_F4, a_gr), (b_F4, b_gr) = old.forward(x), new.forward(x)
    err = max(float((a_F4-b_F4).abs().max()), float((a_gr-b_gr).abs().max()))
    print(f"  forward() max difference over {nbatch} random inputs: {err:.3e}")
    if err != 0.0:
        raise SystemExit(
            "  MISMATCH: the rebuilt network is not the one that was trained.\n"
            "  The weights loaded, so the architecture is right and the likely cause is\n"
            "  an activation, which has no parameters and so loads silently. Re-run with\n"
            "  --scalar-activation / --nonscalar-activation set to what training used.")

    # predict_all differences are expected, but only in the nan -> finite direction
    torch.manual_seed(1)
    F4 = torch.rand(nbatch, 2, int(old.NF), 4)
    F4[..., :3] = (F4[..., :3]-0.5) * 0.5 * F4[..., 3:]
    with torch.inference_mode():
        o_F4, o_gr, _ = old.predict_all(F4)
        n_F4, n_gr, _ = new.predict_all(F4)
    ok_old = torch.isfinite(o_F4).flatten(1).all(1) & torch.isfinite(o_gr)
    ok_new = torch.isfinite(n_F4).flatten(1).all(1) & torch.isfinite(n_gr)
    both = ok_old & ok_new
    d = 0.0 if not bool(both.any()) else float((o_F4[both]-n_F4[both]).abs().max())
    print(f"  predict_all: {int(both.sum())}/{nbatch} finite under both, "
          f"max difference {d:.3e}")
    print(f"               {int((~ok_old & ok_new).sum())} newly finite, "
          f"{int((ok_old & ~ok_new).sum())} newly nan")
    if bool((ok_old & ~ok_new).any()):
        print("  WARNING: the current source refuses points the checkpoint accepted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="existing .pt saved by save_model")
    ap.add_argument("outfilename", help="output prefix; save_model appends _<device>.pt")
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--scalar-activation", default="silu")
    ap.add_argument("--nonscalar-activation", default="sigmoid")
    args = ap.parse_args()

    old = torch.jit.load(args.checkpoint, map_location="cpu")
    old.eval()

    parms = infer_parms(old,
                        resolve_activation(args.scalar_activation),
                        resolve_activation(args.nonscalar_activation))
    print(f"architecture recovered from {args.checkpoint}:")
    for k in ["NF", "tensor_product_class", "nhidden_shared", "irreps_shared",
              "nhidden_growthrate", "nhidden_F4",
              "average_heavies_in_final_state", "conserve_lepton_number"]:
        print(f"  {k:32s} {parms[k]}")
    print(f"  {'activations':32s} {args.scalar_activation} / {args.nonscalar_activation}"
          "  (not stored in the checkpoint - verified below)")

    new = NeuralNetwork(parms)
    # w3j are Wigner 3j constants rebuilt by e3nn, not trained values
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    missing    = [k for k in missing    if "w3j" not in k]
    unexpected = [k for k in unexpected if "w3j" not in k]
    if missing or unexpected:
        raise SystemExit(f"state_dict does not fit the rebuilt model.\n"
                         f"  missing:    {missing[:5]}\n  unexpected: {unexpected[:5]}")
    new.eval()

    print("verifying against the original:")
    verify(old, new)

    for device in args.devices:
        save_model(new, args.outfilename, device)


if __name__ == "__main__":
    main()
