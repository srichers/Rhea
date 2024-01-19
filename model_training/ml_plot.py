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
        assert(len(self.train_loss) >= n_fill_values)

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
    def __init__(self, epochs, names):
        self.data = {}
        for name in names:
            self.data[name] = PlotQuantities(epochs)

    # return the minimum and maximum values of all datasets
    def minmax(self):
        minval = 1e100
        maxval = 0
        for p in self.data.values():
            minval_p, maxval_p = p.minmax()
            minval = min(minval, minval_p)
            maxval = max(maxval, maxval_p)
        return minval, maxval

    # second initializer that will create all of the same quantities, but will also accept a plotter object as an argument
    # and fill in the values from that plotter object
    def fill_from_plotter(self, plotter):
        for name in self.data.keys():
            self.data[name].fill_from_plotquantities(plotter.data[name])

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
        nplots = len(self.data)
        fig,axes=plt.subplots(1,nplots, sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axes.flatten():
            ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
            ax.minorticks_on()

        epochs = len(self.data["knownData"].train_err)
        x = range(epochs)
        
        for i,name in enumerate(self.data.keys()):
            self.plot_error_single_frame(axes[i], x, self.data[name], name)
            axes[i].legend(frameon=False,fontsize=8)
        axes[0].legend(frameon=False,fontsize=8)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Error")
        plt.xlim(0,epochs)

        minval, maxval = self.minmax()
        if ymin==0:
            ymin = minval
        if ymax==0:
            ymax = maxval

        for ax in axes.flatten():
            ax.set_ylim(ymin,ymax)
        
        plt.savefig("train_test_error.pdf",bbox_inches="tight")

def plot_nue_nuebar(model, npoints, nreps,):
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
        F4_pred = torch.tensor(F4_test[None,:,:,:]).float().to(next(model.parameters()).device)
        for j in range(nreps):
            F4_pred = model.predict_F4(F4_pred)
            nee_list_fid[i,j]    = F4_pred[0,3,0,0].to('cpu')
            neebar_list_fid[i,j] = F4_pred[0,3,1,0].to('cpu')
            
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

def plot_histogram(error, bins, xmin, xmax, filename):
    error[np.where(error>xmax)] = xmax
    # plot the histogram
    plt.clf()
    plt.hist(error, bins=bins, range=(xmin,xmax), log=True)
    plt.xlabel("Max Component Error")
    plt.ylabel("Count")
    plt.xlim(xmin,xmax)
    plt.tick_params(axis='both',which="both", direction="in",top=True,right=True)
    plt.minorticks_on()
    plt.savefig(filename,bbox_inches="tight")

# apply the model to a dataset and create a histogram of the magnitudes of the error
# F4 has dimensions [sim, xyzt, nu/nubar, flavor]
def error_histogram(model, F4_initial, F4_final, bins, xmin, xmax, filename):
    # get the predicted final F4
    F4_pred = model.predict_F4(F4_initial)

    # calculate the error
    F4_error = (F4_final - F4_pred).to('cpu').detach().numpy()

    # calculate the magnitude of the error
    F4_error_mag = np.max(np.abs(F4_error), axis=(1,2,3))

    # calculate the total number density in each simulation
    N = F4_final[:,3,:,:].to('cpu').detach().numpy()
    N = np.sum(N, axis=(1,2))

    plot_histogram(F4_error_mag/N, bins, xmin, xmax, filename)

        
# visualize the network using torchviz
#y = model(X_train)
#torchviz.make_dot(y.mean(), params=dict(model.named_parameters())).render("model_torchviz", format="pdf", show_saved=True)
