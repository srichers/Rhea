import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

class PlotQuantities():
    def __init__(self, epochs):
        self.train_loss = np.zeros(epochs)
        self.train_err  = np.zeros(epochs)
        self.test_loss  = np.zeros(epochs)
        self.test_err   = np.zeros(epochs)
    

class Plotter():
    def __init__(self, epochs):
        self.knownData      = PlotQuantities(epochs)
        self.knownData_FS   = PlotQuantities(epochs)
        self.zerofluxfac    = PlotQuantities(epochs)
        self.oneflavor      = PlotQuantities(epochs)
        self.unphysical     = PlotQuantities(epochs)
        self.NSM            = PlotQuantities(epochs)
        self.particlenumber = PlotQuantities(epochs)

    def init_plot_options(self):
        #==============#
        # plot options #
        #==============#
        mpl.rcParams['font.size'] = 8
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

    def plot_error_single_frame(self, ax, x, p, label, ymin, ymax):
        ax.set_title(label, y = 1.0, pad=-14)
        ax.semilogy(x, np.sqrt(p.train_loss), label="Sqrt(train_loss)", color="blue", linewidth=2)
        ax.semilogy(x, np.sqrt(p.test_loss),  label="Sqrt(test_loss)",  color="black", linewidth=2)
        ax.semilogy(x, p.train_err, label="train_maxerr", color="blue", linewidth=0.5)
        ax.semilogy(x, p.test_err,  label="test_maxerr",  color="black", linewidth=0.5)

        # return the minimum and maximum values of all four curves
        this_ymin = min(np.sqrt(p.train_loss).min(), np.sqrt(p.test_loss).min(), p.train_err.min(), p.test_err.min())
        this_ymax = max(np.sqrt(p.train_loss).max(), np.sqrt(p.test_loss).max(), p.train_err.max(), p.test_err.max())
        ymin = min(ymin, this_ymin) if this_ymin > 0 else ymin
        ymax = max(ymax, this_ymax) if this_ymax > 0 else ymax
        return ymin, ymax
        
    def plot_error(self):
        plt.clf()
        fig,axes=plt.subplots(2,3, sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axes.flatten():
            ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
            ax.minorticks_on()

        epochs = len(self.knownData.train_err)
        x = range(epochs)
        
        ymin = 1e100
        ymax = 0
        ymin, ymax = self.plot_error_single_frame(axes[0,0], x, self.knownData,     "known data"         , ymin, ymax)
        ymin, ymax = self.plot_error_single_frame(axes[0,1], x, self.NSM,           "NSM stable"         , ymin, ymax)
        ymin, ymax = self.plot_error_single_frame(axes[0,2], x, self.unphysical,    "unphysical"         , ymin, ymax)
        ymin, ymax = self.plot_error_single_frame(axes[1,0], x, self.knownData_FS,  "final stable"       , ymin, ymax)
        ymin, ymax = self.plot_error_single_frame(axes[1,1], x, self.zerofluxfac,   "zero fluxfac stable", ymin, ymax)
        ymin, ymax = self.plot_error_single_frame(axes[1,2], x, self.oneflavor,     "one flavor stable"  , ymin, ymax)
        axes[0,0].legend(frameon=False,fontsize=8)

        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Error")
        plt.xlim(0,epochs)
        plt.ylim(ymin,ymax)
        
        plt.savefig("train_test_error.pdf",bbox_inches="tight")

    def plot_nue_nuebar(self, model, npoints, nreps,):
        # plot the number of electron neutrinos when varying the number of antineutrinos
        nee_list_fid    = np.zeros((npoints,nreps))
        neebar_list_fid = np.zeros((npoints,nreps))
        nee_list_23     = np.zeros((npoints,nreps))
        neebar_list_23  = np.zeros((npoints,nreps))
        ratio_list = np.array(range(npoints)) / (npoints-1)
        for i in range(npoints):
            F4_test = np.zeros((4,2,model.NF)) # [xyzt, nu/nubar, flavor]
            F4_test[3, 0, 0] =  1
            F4_test[3, 1, 0] =  ratio_list[i]
            F4_test[2, 0, 0] =  1/3
            F4_test[2, 1, 0] = -1/3 * ratio_list[i]
            F4_pred = torch.tensor(F4_test[None,:,:,:]).float()
            for j in range(nreps):
                F4_pred = model.to('cpu').predict_F4(F4_pred.to('cpu'))
                nee_list_fid[i,j]    = F4_pred[0,3,0,0]
                neebar_list_fid[i,j] = F4_pred[0,3,1,0]
                
        plt.clf()
        fig,ax=plt.subplots(1,1)
        ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        ax.minorticks_on()

        for j in range(nreps):
            plt.plot(ratio_list, nee_list_fid[:,j], linewidth=2, color=mpl.cm.jet(j/(nreps-1)) )#, label="$N_{\\nu_e}$")

        plt.legend(frameon=False)
        plt.xlabel("$N_{\\bar{\\nu}_e} / N_{\\nu_e}$")
        plt.ylabel("$N_{\\nu_e}$")
        plt.ylim(0.3,1)
        plt.xlim(0,1)
        plt.axhline(1./3., color="green", linewidth=0.5)
        plt.savefig("nue_vs_nuebar.pdf",bbox_inches="tight")

        
# visualize the network using torchviz
#y = model(X_train)
#torchviz.make_dot(y.mean(), params=dict(model.named_parameters())).render("model_torchviz", format="pdf", show_saved=True)
