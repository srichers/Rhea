# Given a full set of simulations sorted into [a-z]*10/reduced0D.h5, extract a database of the FFI growth rate and saturation properties
# Run from within a single RUN directory
import os
import h5py
import re
import numpy as np
from scipy.signal import argrelextrema
from multiprocessing import Pool
import generate
import maxentropy
import torch

NF = 3
file_format = "hdf5" # "hdf5" or "dat"
growthrate_minfac = 1e-3
growthrate_maxfac = 1e-1
nproc = 32
average_start_tsat = 2 # start averaging at 2*tsat
average_min_tsat = 3 # require time up to at least 3*tsat
problematic_stddev_val = 0.05
violation_limit = 1e-5
#problematic_final_val = 20

#=================================================#
# read in the stable points from the NSM snapshot #
#=================================================#
def write_NSM_stable_data(filename, outfilename):
    # note that x represents the SUM of mu, tau, anti-mu, anti-tau and must be divided by 4 to get the individual flavors
    # take only the y-z slice to limit the size of the data.
    f_in = h5py.File(filename,"r")
    discriminant = torch.from_numpy(f_in["crossing_discriminant"][...]).flatten()
    # n has shape [Nx,Ny,Nz]]
    ne = torch.from_numpy(f_in["n_e(1|ccm)"][...]).flatten()
    na = torch.from_numpy(f_in["n_a(1|ccm)"][...]).flatten()
    nx = torch.from_numpy(f_in["n_x(1|ccm)"][...]).flatten()
    # f has shape [3, Nx,Ny,Nz]
    fe = torch.from_numpy(f_in["fn_e(1|ccm)"][...]).flatten(start_dim=1)
    fa = torch.from_numpy(f_in["fn_a(1|ccm)"][...]).flatten(start_dim=1)
    fx = torch.from_numpy(f_in["fn_x(1|ccm)"][...]).flatten(start_dim=1)
    f_in.close()

    num_elements = discriminant.numel()
    print("NSM number of elements:",num_elements)
    
    # set stability bit
    stable = torch.zeros(num_elements)
    stable_locs = torch.where(discriminant<=0)
    unstable_locs = torch.where(discriminant>0)
    print(len(stable_locs[0]),"stable points")
    print(len(unstable_locs[0]),"unstable points")
    stable[stable_locs] = 1

    F4_NSM = torch.zeros((num_elements,4,2,NF))
    F4_NSM[:,3,0,0  ] = ne
    F4_NSM[:,3,1,0  ] = na
    F4_NSM[:,3,:,1:3] = nx[:,None,None] / 4.
    for i in range(3):
        F4_NSM[:,i,0,0  ] = fe[i]
        F4_NSM[:,i,1,0  ] = fa[i]
        F4_NSM[:,i,:,1:3] = fx[i][:,None,None] / 4.

    generate.write_stable_dataset(outfilename, F4_NSM, stable)


# get the list of files to process
def simulation_is_finished(dirname):
    file_to_check = dirname+"/output.txt"
    if os.path.exists(file_to_check) and "finalized" in open(file_to_check).readlines()[-1]:
        return True
    else:
        return False

#========================#
# get the grid cell size #
#========================#
def cell_size(dirname):
    f = open(dirname+"/inputs","r")
    lines = f.readlines()
    for line in lines:
        if line[:2] == "Lz":
            Lz = float(line.split("=")[1])
        if line[:5] == "ncell":
            nz = int(re.split("[, \)]", line)[-2])
    f.close()
    return Lz/nz


#==========================#
# get the growth rate, etc #
#==========================#
def growth_properties(data):
    t = np.array(data["time(s)"]) # seconds
    N_offdiag_mag = np.array(data["N_offdiag_mag(1|ccm)"])

    # get the time of the peak
    flag = True                                                                # Flag to prevent any errors if the following method fails
    indexes = np.concatenate([[0],argrelextrema(N_offdiag_mag,np.greater)[0]]) # Take all the indexes at which there is a local maxima, as well as the first point in the data
    for i in range(len(indexes)-1):                                            # Loops over all the indexes to check if there are two adjacent points that have a difference of three orders of magnitude
        if abs(round((np.log10(N_offdiag_mag[indexes[i]]/N_offdiag_mag[indexes[i+1]])))) >= 3:
            imax = indexes[i+1]
            flag = False
    if flag == True:                                                           # If the previous method does not work, the following is used
        imax = np.argmax(N_offdiag_mag)

    # get the growth rate
    minOffDiag = N_offdiag_mag[0] #np.min(Nbar_offdiag_mag)
    maxOffDiag = np.max(N_offdiag_mag)

    # calculate the growth rate
    growthRate = 0
    locs  = np.where((N_offdiag_mag > growthrate_minfac*maxOffDiag) & (N_offdiag_mag < growthrate_maxfac*maxOffDiag) & (t<t[imax]))[0]
    if len(locs)>1:
        i0 = locs[0]
        i1 = locs[-1]
        growthRate = np.log(N_offdiag_mag[i1]/N_offdiag_mag[i0]) / (t[i1]-t[i0])
            
    return imax, growthRate

def bad_data(F4_initial, F4_final, F4_final_stddev):
    Ntot_initial = np.sum(F4_initial[3,:,:],axis=(0,1))

    relative_stddev = np.max(F4_final_stddev/Ntot_initial)
    if np.any(relative_stddev>problematic_stddev_val):
        print("   Warning: large standard deviation",relative_stddev)
        return True
    
    F_violation = np.max(np.abs(np.sum(F4_initial-F4_final, axis=2))) / Ntot_initial
    if F_violation > violation_limit:
        print("   Warning: large F violation",F_violation)
        return True

    ELN_initial = F4_initial[3,0,:]-F4_initial[3,1,:]
    ELN_final   =   F4_final[3,0,:]-  F4_final[3,1,:]
    ELN_violation = np.max(np.abs(ELN_final-ELN_initial)) / Ntot_initial
    if ELN_violation > violation_limit:
        print("   Warning: large ELN violation",ELN_violation)
        return True

    return False

#===========================================#
# get the final moments of the distribution #
#===========================================#
# divide by dz because dx and dy are assumed to be 1
# each result has dimensions [xyzt, nu/nubar, flavor]
def final_properties(imax, growthRate, data, dz):
    t = np.array(data["time(s)"]) # seconds
    nt = len(t) 
    
    # [xyzt, nu/antinu, flavor, it]
    F4 = np.zeros((4,2,NF,nt))
    for ixyzt, xyzt in zip(range(4),["Fx","Fy","Fz","N"]):
        for isuffix, suffix in zip(range(2),["","bar"]):
            for flavor in range(NF):
                F4[ixyzt,isuffix,flavor,:] = np.array(data[xyzt+str(flavor)+str(flavor)+suffix+"(1|ccm)"])

    # get the time interval over which to average
    Ntot = np.sum(F4[3,:,:,0],axis=(0,1))
    F4_initial = F4[:,:,:,0]
    F4_final = F4_initial * np.NAN
    F4_final_stddev = F4_initial * np.NAN
    tmax = t[imax]
    tend = t[-1]

    xplot = t/tmax
    y0plot = F4[3,0,0,:] / F4[3,0,0,0]
    y1plot = np.array(data["N_offdiag_mag(1|ccm)"]) / Ntot
    
    if tend > average_min_tsat*tmax and growthRate>0:

        i0 = np.argmin(np.abs(t-average_start_tsat*tmax))
        i1 = -1 #np.argmin(np.abs(t-average_min_tsat*tmax))
        F4_final        = np.average(F4[:,:,:,i0:i1], axis=(3))
        F4_final_stddev = np.std(    F4[:,:,:,i0:i1], axis=(3))

        if bad_data(F4_initial, F4_final, F4_final_stddev):
            F4_initial *= np.NAN
            F4_final_stddev *= np.NAN
            F4_final *= np.NAN

    return F4_initial, F4_final, F4_final_stddev, xplot, y0plot, y1plot


def analyze_file(d):
    #print(d)
    # get the cell size
    dz = cell_size(d)
    
    # open the data file
    if file_format == "hdf5":
        data = h5py.File(d+"/reduced0D.h5","r")
    elif file_format == "dat":
        with open(d+"/reduced0D.dat","r") as f:
            headers = f.readline().split()
            keys = [h.split(":")[1] for h in headers]
        input_data = np.genfromtxt(d+"/reduced0D.dat",skip_header=1).transpose()

        data = {}
        for i,k in enumerate(keys):
            data[k] = input_data[i]

    # get the growth rate
    imax, growthRate = growth_properties(data)
    
    # get the final state
    F4_initial, F4_final, F4_final_stddev, xplot, y0plot, y1plot = final_properties(imax, growthRate, data, dz)

    # close the data file
    if file_format == "hdf5":
        data.close()

    # return data
    return growthRate, F4_initial, F4_final, F4_final_stddev, xplot, y0plot, y1plot

def write_asymptotic_dataset(directory, outfilename):

    #=============================#
    # get the list of directories #
    #=============================#
    print(directory)
    dirlist = sorted([ f.path for f in os.scandir(directory) if f.is_dir() ])
    finished_list = [f for f in dirlist if simulation_is_finished(f)]
    print(len(dirlist),"directories")
    
    nsims = len(finished_list)
    print(nsims,"finished simulations")
    
    # get the number of flavors
    data = h5py.File(finished_list[0]+"/reduced0D.h5","r")
    if "N22(1|ccm)" in data.keys(): nf = 3
    else: nf = 2
    data.close()
    print(str(nf) + " flavors")
    assert(nf==NF)
    
    # get the list of finished simulations
    print("Total finished list length:",len(finished_list))
    #finished_list = finished_list[:10]

    
    #=================#
    # LOOP OVER FILES #
    #=================#
    with Pool(nproc) as p:
        results = p.map(analyze_file, finished_list)
    #results = []
    #for p in finished_list:
    #    results.append(analyze_file(p))

    # arrange data into arrays
    growthRateList    = np.array([r[0] for r in results])
    F4InitialList     = np.array([r[1] for r in results])
    F4FinalList       = np.array([r[2] for r in results])
    F4FinalStddevList = np.array([r[3] for r in results])
    xplotList         = np.array([r[4] for r in results])
    y0plotList        = np.array([r[5] for r in results])
    y1plotList        = np.array([r[6] for r in results])
    
    # count the number of good results, defined by F4_final not being nan
    print("Total number of input simulations:",F4FinalList.shape[0])
    goodlocs = np.where(F4FinalList[:,0,0,0]==F4FinalList[:,0,0,0])[0]
    print("Number of good results:",len(goodlocs))
    
    # select only the good points for the dataset
    print("Number of points before selection:",growthRateList.shape[0])
    growthRateList    = growthRateList[goodlocs]
    F4InitialList     = F4InitialList[goodlocs,:,:,:]
    F4FinalList       = F4FinalList[goodlocs,:,:,:]
    F4FinalStddevList = F4FinalStddevList[goodlocs,:,:,:]
    xplotList         = xplotList[goodlocs,:]
    y0plotList        = y0plotList[goodlocs,:]
    y1plotList        = y1plotList[goodlocs,:]
    print("Number of points after selection:",growthRateList.shape[0])
    
    # Write data to hdf5 file
    output = h5py.File(outfilename,"w")
    output["nf"] = NF
    output["growthRate(1|s)"] = growthRateList
    output["F4_final(1|ccm)"] = F4FinalList
    output["F4_final_stddev(1|ccm)"] = F4FinalStddevList
    output["F4_initial(1|ccm)"] = F4InitialList
    output["xplot"] = xplotList
    output["y0plot"] = y0plotList
    output["y1plot"] = y1plotList
    output["directorynames"] = np.array(finished_list, dtype=object)[goodlocs]
    output.close()
    print()

if __name__ == "__main__":
    # stable/unstable criteria based on maximum entropy conditions applied to NSM snapshot data
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl0_orthonormal.h5","stable_M1-NuLib_rl0.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl1_orthonormal.h5","stable_M1-NuLib_rl1.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl2_orthonormal.h5","stable_M1-NuLib_rl2.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl3_orthonormal.h5","stable_M1-NuLib_rl3.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-LeakageRates/M1VolumeData/M1VolumeData/model_rl0_orthonormal.h5","stable_M1-LeakageRates_rl0.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-LeakageRates/M1VolumeData/M1VolumeData/model_rl1_orthonormal.h5","stable_M1-LeakageRates_rl1.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-LeakageRates/M1VolumeData/M1VolumeData/model_rl2_orthonormal.h5","stable_M1-LeakageRates_rl2.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-LeakageRates/M1VolumeData/M1VolumeData/model_rl3_orthonormal.h5","stable_M1-LeakageRates_rl3.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib-old/orthonormal_distributions/model_rl0_orthonormal.h5","stable_M1-NuLib-old_rl0.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib-7ms/model_rl0_orthonormal.h5","stable_M1-Nulib-7ms_rl0.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib-7ms/model_rl1_orthonormal.h5","stable_M1-Nulib-7ms_rl1.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib-7ms/model_rl2_orthonormal.h5","stable_M1-Nulib-7ms_rl2.h5")
    #write_NSM_stable_data("/mnt/scratch/NSM_ML/spec_data/M1-NuLib-7ms/model_rl3_orthonormal.h5","stable_M1-Nulib-7ms_rl3.h5")

    # randomly generated distributions
    ngenerate = 1000
    nphi_maxentropy_check = 128

    result = generate.generate_stable_F4_zerofluxfac(NF, ngenerate, False).numpy()
    print("generate_stable_F4_zerofluxfac output: ",result.shape)
    generate.write_stable_dataset("stable_zerofluxfac.h5",result, torch.ones(ngenerate))
    
    result = generate.generate_stable_F4_oneflavor(NF, ngenerate, False).numpy()
    print("generate_stable_F4_oneflavor output: ",result.shape)
    generate.write_stable_dataset("stable_oneflavor.h5", result, torch.ones(result.shape[0]))

    result = generate.generate_random_F4(NF, ngenerate, False, 10, 0.95).numpy()
    hascrossing = torch.tensor(maxentropy.has_crossing(result, 3, nphi_maxentropy_check))
    print("generate_stable_F4_oneflavor output: ",result.shape)
    print("nstable:",torch.sum(hascrossing))
    generate.write_stable_dataset("stable_random.h5", result, 1-hascrossing)

    # datasets from many Emu simulations
    #write_asymptotic_dataset("/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-old", "asymptotic_M1-NuLib-old.h5")
    #write_asymptotic_dataset("/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib", "asymptotic_M1-NuLib.h5")
    #write_asymptotic_dataset("/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-7ms", "asymptotic_M1-NuLib-7ms.h5")
    #write_asymptotic_dataset("/mnt/scratch/NSM_ML/Emu_merger_grid/maximum_entropy_32beam_effective2flavor", "asymptotic_random.h5")
    
