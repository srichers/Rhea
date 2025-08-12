'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file that is actually run to train a model. It requires access to various databases that are published elsewhere. All model hyperparameters are listed here.
'''

if __name__ == "__main__":
    import numpy as np
    import torch
    from torch import nn

    from ml_loss import *
    from ml_neuralnet import *
    from ml_optimizer import *
    from ml_plot import *
    from ml_trainmodel import *
    from ml_maxentropy import *
    from ml_read_data import *
    from ml_tools import *
    import pickle
    import torch.optim
    import torch.autograd.profiler as profiler

    # create a list of options
    parms = {}

    parms["database_list"] = [
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-old/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-7ms/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/maximum_entropy_32beam_effective2flavor/many_sims_database.h5"
    ]
    parms["do_unpickle"] = False
    parms["unpickle_filename"] = None
    parms["test_size"] = 0.1
    parms["epochs"] = 150000
    parms["dataset_size_list"] = [-1] # -1 means use all the data
    parms["n_generate"] = 200000
    parms["print_every"] = 10
    parms["output_every"] = 5000
    parms["generate_max_fluxfac"] = 0.95
    parms["generate_zero_weight"] = 10
    parms["average_heavies_in_final_state"] = True
    parms["conserve_lepton_number"] = "direct"

    # data augmentation options
    parms["do_augment_permutation"]=False # this is the most expensive option to make true, and seems to make things worse...
    parms["do_augment_final_stable"]= False # True
    parms["do_unphysical_check"]= True # True - seems to help prevent crazy results
    parms["do_augment_0ff"]= True
    parms["do_augment_random_stable"]= True
    parms["do_augment_NSM_stable"]= True

    # neural network options
    parms["nhidden"]= 3
    parms["width"]= 128
    parms["dropout_probability"]= 0.1 #0.1 #0.5 #0.1 # 0.5
    parms["do_batchnorm"]= False # False - Seems to make things worse
    parms["do_fdotu"]= True
    parms["activation"]= nn.LeakyReLU # nn.LeakyReLU, nn.ReLU

    # optimizer options
    parms["op"]= torch.optim.AdamW # Adam, SGD, RMSprop
    parms["amsgrad"]= False
    parms["weight_decay"]= 1e-2 #1e-5
    parms["learning_rate"]= 2e-4 # 1e-3
    parms["fused"]= True
    parms["lr_scheduler"]= torch.optim.lr_scheduler.ReduceLROnPlateau
    parms["patience"]= 500
    parms["cooldown"]= 500
    parms["factor"]= 0.5
    parms["min_lr"]= 0 #1e-8

    # the number of flavors should be 3
    parms["NF"]= 3

    #========================#
    # use a GPU if available #
    #========================#
    parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using",parms["device"],"device")
    print(torch.cuda.get_device_name(0))

    #===============#
    # read the data #
    #===============#
    if parms["do_unpickle"]:
        with open("train_test_datasets.pkl", "rb") as f:
            F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test = pickle.load(f)
    else:
        F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test = read_test_train_data(parms)

    # move the arrays over to the gpu
    F4i_train = torch.Tensor(F4i_train).to(parms["device"])
    F4f_train = torch.Tensor(F4f_train).to(parms["device"])
    F4i_test  = torch.Tensor(F4i_test ).to(parms["device"])
    F4f_test  = torch.Tensor(F4f_test ).to(parms["device"])
    print("Train:",F4i_train.shape)
    print("Test:",F4i_test.shape)

    # adjust entries of -1 to instead have the correct size of the dataset
    for i in range(len(parms["dataset_size_list"])):
        if parms["dataset_size_list"][i] == -1:
            parms["dataset_size_list"][i] = F4i_train.shape[0]

    #=======================#
    # instantiate the model #
    #=======================#
    print()
    print("#############################")
    print("# SETTING UP NEURAL NETWORK #")
    print("#############################")
    # set up an array of models, optimizers, and plotters for different dataset sizes
    model_array = []
    optimizer_array = []
    plotter_array = []
    scheduler_array = []

    for dataset_size in parms["dataset_size_list"]:
        if parms["do_unpickle"]:
            with open(parms["unpickle_filename"], "rb") as f:
                model, optimizer, plotter = pickle.load(f)

        else:
            model = AsymptoticNeuralNetwork(parms, None).to(parms["device"]) #nn.Tanh()
            plotter = Plotter(0,["ndens","fluxmag","direction","logGrowthRate","unphysical"])

        plotter_array.append(plotter)
        model_array.append(model)
        optimizer_array.append(Optimizer(
            model_array[-1],
            parms["op"](model.parameters(),
                        weight_decay=parms["weight_decay"],
                        lr=parms["learning_rate"],
                        amsgrad=parms["amsgrad"],
                        fused=parms["fused"]),
            parms["device"]))
        scheduler_array.append(parms["lr_scheduler"](optimizer_array[-1].optimizer,
                                                    patience=parms["patience"],
                                                    cooldown=parms["cooldown"],
                                                    factor=parms["factor"],
                                                    min_lr=parms["min_lr"])) # 

    print(model_array[-1])
    print("number of parameters:", sum(p.numel() for p in model_array[-1].parameters()))

    print()
    print("######################")
    print("# Training the model #")
    print("######################")
    #with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
    for i in range(len(parms["dataset_size_list"])):
        model_array[i], optimizer_array[i], scheduler_array[i], plotter_array[i] = train_asymptotic_model(
            parms,
            model_array[i],
            optimizer_array[i],
            scheduler_array[i],
            plotter_array[i],
            parms["dataset_size_list"][i],
            F4i_train,
            F4f_train,
            F4i_test,
            F4f_test,
            logGrowthRate_train,
            logGrowthRate_test)

        # pickle the model, optimizer, and plotter
        with open("model_"+str(parms["dataset_size_list"][i])+".pkl", "wb") as f:
            pickle.dump([model_array[i], optimizer_array[i], plotter_array[i]], f)
