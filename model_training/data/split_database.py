'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file splits an asymptotic database into chunks that can be used separately as training,
validation, and test data. Points with too large a flux factor are removed, the survivors are
divided into contiguous chunks, and each chunk may be thinned to a lower sampling density. The
databases are stored sorted by grid index, so a contiguous chunk of rows is a contiguous region of
the simulation. The parameters used are recorded in the output filenames and file attributes.
'''

import os
import sys
import argparse
import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_tools as ml

# datasets holding one entry per simulation, which are subset along with the points
per_sim_datasets = ["F4_initial(1|ccm)",
                    "F4_final(1|ccm)",
                    "F4_final_stddev(1|ccm)",
                    "growthRate(1|s)",
                    "directorynames"]

# datasets describing the database as a whole, copied through unchanged
whole_database_datasets = ["nf"]

# per-simulation time series used only for plotting. Nothing in the training path reads these and
# they are roughly 99.9% of the file size, so they are left behind in the source database.
dropped_datasets = ["xplot", "y0plot", "y1plot"]

# remove any point reaching max_fluxfac in any nu/nubar/flavor species. This is the cut
# ml_read_data applies at load time, with the threshold exposed as a parameter.
# input dimensions: [sim, xyzt, nu/nubar, flavor]
# output dimensions: [point]
def capped_indices(F4_initial, max_fluxfac):
    # ml_tools expects [sim, nu/nubar, flavor, xyzt]
    fluxfac = ml.flux_factor(torch.Tensor(F4_initial).permute(0,2,3,1))
    return torch.where(torch.all(fluxfac < max_fluxfac, dim=(1,2)))[0].numpy()

def split_database(infilename, outdirectory, max_fluxfac, n_chunks, thinning):
    print()
    print("#",infilename)

    #===================#
    # read the database #
    #===================#
    data = {}
    with h5py.File(infilename,"r") as f_in:
        for key in f_in.keys():
            if key in dropped_datasets:
                continue
            assert(key in per_sim_datasets or key in whole_database_datasets)
            if key=="directorynames":
                data[key] = f_in[key].asstr()[...]
            else:
                data[key] = f_in[key][...]
    npoints_source = len(data["F4_initial(1|ccm)"])
    print("#   ",npoints_source,"points in source database")

    #===============================================#
    # cap the flux factor before dividing the data, #
    # so the chunks hold equal surviving counts     #
    #===============================================#
    goodlocs = capped_indices(data["F4_initial(1|ccm)"], max_fluxfac)
    for key in data.keys():
        if key not in whole_database_datasets:
            data[key] = data[key][goodlocs]
    npoints_capped = len(goodlocs)
    print("#   ",npoints_capped,"points with flux factor below",max_fluxfac)
    assert(npoints_capped > 0)

    #=========================================#
    # divide into contiguous chunks and write #
    #=========================================#
    basename = os.path.basename(infilename)[:-len(".h5")]
    for ichunk, chunk in enumerate(np.array_split(np.arange(npoints_capped), n_chunks)):
        # keep the points whose chunk-local index is a multiple of thinning
        indices = chunk[::thinning]
        assert(len(indices) > 0)

        # chunk2-0 means the first of two chunks
        suffix      = "_chunk"+str(n_chunks)+"-"+str(ichunk) + "_thin"+str(thinning) + "_maxfluxfac"+("%g"%max_fluxfac)
        outfilename = os.path.join(outdirectory, basename+suffix+".h5")
        print("#    ",len(indices),"points ->",outfilename)

        with h5py.File(outfilename,"w") as f_out:
            for key in data.keys():
                if key in whole_database_datasets:
                    f_out[key] = data[key]
                else:
                    f_out[key] = data[key][indices]
            f_out.attrs["source_file"]        = infilename
            f_out.attrs["max_fluxfac"]        = max_fluxfac
            f_out.attrs["n_chunks"]           = n_chunks
            f_out.attrs["chunk_index"]        = ichunk
            f_out.attrs["thinning"]           = thinning
            f_out.attrs["n_points_source"]    = npoints_source
            f_out.attrs["n_points_after_cap"] = npoints_capped
            f_out.attrs["n_points_written"]   = len(indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cap, chunk, and thin asymptotic databases.")
    parser.add_argument("infilenames", nargs="+",
                        help="asymptotic database(s) to split")
    parser.add_argument("--max_fluxfac", type=float, default=0.9,
                        help="drop points whose initial flux factor reaches this value")
    parser.add_argument("--n_chunks", type=int, default=2,
                        help="number of contiguous chunks to divide each database into")
    parser.add_argument("--thinning", type=int, default=1,
                        help="keep one of every this many points within each chunk")
    parser.add_argument("--outdirectory", default=None,
                        help="where to write the chunks (default: alongside the input file)")
    args = parser.parse_args()

    assert(args.n_chunks > 0)
    assert(args.thinning > 0)
    assert(args.max_fluxfac > 0)

    for infilename in args.infilenames:
        outdirectory = args.outdirectory
        if outdirectory is None:
            outdirectory = os.path.dirname(os.path.abspath(infilename))
        split_database(infilename, outdirectory, args.max_fluxfac, args.n_chunks, args.thinning)
