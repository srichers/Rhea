import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

NF = 3

f = h5py.File("reduced0D.h5","r")
t = np.array(f["time(s)"])

suffix=["","bar"]
N = np.zeros((2,NF,len(t)))
for i in range(2):
    for j in range(NF):
        datasetname = "N"+str(j)+str(j)+suffix[i]+"(1|ccm)"
        N[i,j,:] = np.array(f[datasetname])
f.close()

#==============#
# plot options #
#==============#
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2

fig,axes=plt.subplots(1,3)
plt.subplots_adjust(wspace=0, hspace=0)
fig.align_labels()

for j in range(NF):
    axes[j].plot(t,N[0,j,:])
    axes[j].plot(t,N[1,j,:])
    axes[j].plot(t,N[0,j,:] - N[1,j,:])

plt.savefig("quickplot.pdf",bbox_inches="tight")
