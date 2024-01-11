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

    # second initializer that will create all of the same quantities,
    # but will also accept a plotquantities object of a smaller size
    # and fill in the values from that plotquantities object
    def fill_from_plotquantities(self, plotquantities):
        n_fill_values = len(plotquantities.train_loss)
        assert(len(self.train_loss) > n_fill_values)

        self.train_loss[:n_fill_values] = plotquantities.train_loss
        self.train_err[ :n_fill_values] = plotquantities.train_err
        self.test_loss[ :n_fill_values] = plotquantities.test_loss
        self.test_err[  :n_fill_values] = plotquantities.test_err

    # return the minimum of only nonzero values, and the maximum of all datasets
    def minmax(self):
        minval = 1e100
        maxval = 0
        for p in [self.train_loss, self.train_err, self.test_loss, self.test_err]:
            nonzero_values = p[np.where(p>0)]
            if len(nonzero_values) > 0:
                minval = min(minval, nonzero_values.min())
                maxval = max(maxval, nonzero_values.max())
        return minval, maxval
    

class Plotter():
    def __init__(self, epochs):
        self.knownData      = PlotQuantities(epochs)
        self.knownData_FS   = PlotQuantities(epochs)
        self.zerofluxfac    = PlotQuantities(epochs)
        self.oneflavor      = PlotQuantities(epochs)
        self.unphysical     = PlotQuantities(epochs)
        self.NSM            = PlotQuantities(epochs)

    # return the minimum and maximum values of all datasets
    def minmax(self):
        minval = 1e100
        maxval = 0
        for p in [self.knownData, self.knownData_FS, self.zerofluxfac, self.oneflavor, self.unphysical, self.NSM]:
            minval_p, maxval_p = p.minmax()
            minval = min(minval, minval_p)
            maxval = max(maxval, maxval_p)
        return minval, maxval

    # second initializer that will create all of the same quantities, but will also accept a plotter object as an argument
    # and fill in the values from that plotter object
    def fill_from_plotter(self, plotter):
        self.knownData.fill_from_plotquantities(plotter.knownData)
        self.knownData_FS.fill_from_plotquantities(plotter.knownData_FS)
        self.zerofluxfac.fill_from_plotquantities(plotter.zerofluxfac)
        self.oneflavor.fill_from_plotquantities(plotter.oneflavor)
        self.unphysical.fill_from_plotquantities(plotter.unphysical)
        self.NSM.fill_from_plotquantities(plotter.NSM)

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

    def plot_error_single_frame(self, ax, x, p, label):
        ax.set_title(label, y = 1.0, pad=-14)
        ax.semilogy(x, np.sqrt(p.train_loss), label="Sqrt(train_loss)", color="blue", linewidth=2)
        ax.semilogy(x, np.sqrt(p.test_loss),  label="Sqrt(test_loss)",  color="black", linewidth=2)
        ax.semilogy(x, p.train_err, label="train_maxerr", color="blue", linewidth=0.5)
        ax.semilogy(x, p.test_err,  label="test_maxerr",  color="black", linewidth=0.5)

    def plot_error(self, ymin=0, ymax=0):
        plt.clf()
        fig,axes=plt.subplots(2,3, sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axes.flatten():
            ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
            ax.minorticks_on()

        epochs = len(self.knownData.train_err)
        x = range(epochs)
        
        self.plot_error_single_frame(axes[0,0], x, self.knownData,     "known data"         )
        self.plot_error_single_frame(axes[0,1], x, self.NSM,           "NSM stable"         )
        self.plot_error_single_frame(axes[0,2], x, self.unphysical,    "unphysical"         )
        self.plot_error_single_frame(axes[1,0], x, self.knownData_FS,  "final stable"       )
        self.plot_error_single_frame(axes[1,1], x, self.zerofluxfac,   "zero fluxfac stable")
        self.plot_error_single_frame(axes[1,2], x, self.oneflavor,     "one flavor stable"  )
        axes[0,0].legend(frameon=False,fontsize=8)

        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Error")
        plt.xlim(0,epochs)

        minval, maxval = self.minmax()
        if ymin==0:
            ymin = minval
        if ymax==0:
            ymax = maxval

        for ax in axes.flatten():
            ax.set_ylim(ymin,ymax)
        
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
