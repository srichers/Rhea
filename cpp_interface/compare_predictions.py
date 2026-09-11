#!/usr/bin/env python3
'''
Authors: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

Compares the predictions written by predict_torch and predict_kokkos. The two consumers
share no code and are never linked together - the LibTorch side reads the .pt directly and
the Kokkos side reads a .rhea written by kokkos/export_rhea.py - so agreement here
is a real check on the export rather than a tautology.

Exact agreement is not expected: the two sum the same terms in different orders in float32,
on different devices. The tolerance is set an order of magnitude above what is measured so
a backend's reassociation does not trip it, while a formula error cannot hide under it.

Usage:
    python3 compare_predictions.py torch.bin kokkos.bin
'''
import argparse
import struct
import sys

import numpy as np

MAGIC     = b"RHEAPRED"
TOLERANCE = 1e-4

def load(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:8] == MAGIC, f"{path} is not a prediction file"
    n = struct.unpack_from("<i", d, 8)[0]
    off = 12
    def take(count, shape):
        nonlocal off
        a = np.frombuffer(d, dtype="<f4", count=count, offset=off).reshape(shape)
        off += 4*count
        return a
    out = {"ncells": n,
           "F4_in":      take(n*24, (n,2,3,4)),
           "F4_out":     take(n*24, (n,2,3,4)),
           "growthrate": take(n,    (n,)),
           "stability":  take(n,    (n,))}
    assert off == len(d), (off, len(d))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="predictions from one consumer")
    ap.add_argument("b", help="predictions from the other")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    assert A["ncells"] == B["ncells"], (A["ncells"], B["ncells"])
    n = A["ncells"]

    # the two generate their own cells from the same deterministic seed, so this catches
    # a divergence in the generator before it can be mistaken for a divergence in the model
    if not np.array_equal(A["F4_in"], B["F4_in"]):
        print("#  FAIL: the two ran on different cells")
        return 1
    print(f"#  {n} cells, identical inputs on both sides")

    nan_a, nan_b = np.isnan(A["growthrate"]), np.isnan(B["growthrate"])
    agree = int((nan_a == nan_b).sum())
    print(f"#  nan pattern agrees on {agree}/{n} cells ({int(nan_a.sum())} nan from box3d)")

    ok = ~nan_a & ~nan_b
    st_mismatch = int((A["stability"][ok] != B["stability"][ok]).sum())
    print(f"#  stability mismatches: {st_mismatch}")

    # F4_out spans four decades across the batch, so scale each cell by its own largest
    # component rather than by a batch-wide maximum
    scale = np.abs(A["F4_out"][ok]).reshape(-1,24).max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    eF4 = float((np.abs(A["F4_out"][ok] - B["F4_out"][ok]).reshape(-1,24).max(axis=1)/scale).max())

    # growthrate is ntot times an O(1) quantity and passes through zero at marginal
    # stability, so a plain relative error is meaningless there. Scale by ntot, which is
    # the scale the model works on internally.
    ntot = A["F4_in"][ok][...,3].sum(axis=(1,2))
    dgr  = np.abs(A["growthrate"][ok] - B["growthrate"][ok])/ntot
    egr, worst = float(dgr.max()), int(np.argmax(dgr))

    print(f"#  F4_out     max relative difference {eF4:.3e}")
    print(f"#  growthrate max difference / ntot   {egr:.3e}  (worst of the finite cells: {worst})")

    status = 0
    if agree != n:
        print("#  FAIL: the two disagree on which cells box3d refuses")
        status = 1
    if st_mismatch != 0:
        print("#  FAIL: stability differs")
        status = 1
    if not eF4 < args.tolerance:
        print(f"#  FAIL: F4_out beyond {args.tolerance:.0e}")
        status = 1
    if not egr < args.tolerance:
        print(f"#  FAIL: growthrate beyond {args.tolerance:.0e}")
        status = 1
    if status == 0:
        print(f"#  PASS: the two agree to better than {args.tolerance:.0e}")
    return status

if __name__ == "__main__":
    sys.exit(main())
