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
    from ml_plot import *
    from ml_trainmodel import *
    from ml_read_data import *
    from ml_tools import *
    import torch.optim
    import torch.autograd.profiler as profiler

    # create a list of options
    parms = {}

    parms["database_list"] = [
        "data/dummy_asymptotic.h5",
    ]
    parms["stable_database_list"] = [
        "data/stable_oneflavor.h5",
        "data/stable_random.h5",
        "data/stable_zerofluxfac.h5",
    ]
    parms["test_size"] = 0.1
    parms["epochs"] = 10
    parms["print_every"] = 1
    parms["output_every"] = 10
    parms["average_heavies_in_final_state"] = False
    parms["conserve_lepton_number"] = "direct"
    parms["random_seed"] = 42
    parms["batch_size"] = 32
    parms["epoch_num_samples"] = 1000
    
    # data augmentation options
    parms["do_augment_permutation"]=False # this is the most expensive option to make true, and seems to make things worse...
    parms["do_augment_final_stable"]= False # True
    parms["do_unphysical_check"]= True # True - seems to help prevent crazy results

    # neural network options
    parms["nhidden_shared"]        = 1
    parms["nhidden_stability"]     = 2
    parms["nhidden_growthrate"] = 2
    parms["nhidden_asymptotic"]    = 2
    parms["nhidden_density"]       = 2
    parms["nhidden_flux"]          = 2
    parms["width_shared"]        = 128
    parms["width_stability"]     = 64
    parms["width_growthrate"] = 64
    parms["width_density"]       = 64
    parms["width_flux"]          = 64
    parms["dropout_probability"]= 0.1 #0.1 #0.5 #0.1 # 0.5
    parms["do_batchnorm"]= False
    parms["do_fdotu"]= True
    parms["activation"]= nn.LeakyReLU # nn.LeakyReLU, nn.ReLU

    # optimizer options
    parms["op"]= torch.optim.AdamW # Adam, SGD, RMSprop
    parms["amsgrad"]= False
    parms["weight_decay"]= 1e-2 #1e-5
    parms["learning_rate"]= 2e-4 # 1e-3
    parms["fused"]= True
    parms["patience"]= 500
    parms["cooldown"]= 500
    parms["factor"]= 0.5
    parms["warmup_iters"]=10
    parms["min_lr"]= 0 #1e-8

    # the number of flavors should be 3
    parms["NF"]= 3

    #========================#
    # use a GPU if available #
    #========================#
    parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using",parms["device"],"device")
    if parms["device"] == "cuda":
        print(torch.cuda.get_device_name(0))

    #===============#
    # read the data #
    #===============#
    dataset_asymptotic_train_list, dataset_asymptotic_test_list = read_asymptotic_data(parms)
    dataset_stable_train_list, dataset_stable_test_list = read_stable_data(parms)

    #=======================#
    # instantiate the model #
    #=======================#
    print()
    print("#############################")
    print("# SETTING UP NEURAL NETWORK #")
    print("#############################")
    model = NeuralNetwork(parms).to(parms["device"]) #nn.Tanh()
    plotter = Plotter(parms["epochs"],["ndens","fluxmag","direction","growthrate","stability","unphysical"])
    optimizer = parms["op"](model.parameters(),
                            weight_decay=parms["weight_decay"],
                            lr=parms["learning_rate"],
                            amsgrad=parms["amsgrad"],
                            fused=parms["fused"]
    )

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=1.0/parms["warmup_iters"],
                                                         end_factor=1,
                                                         total_iters=parms["warmup_iters"])
    scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                patience=parms["patience"],
                                                                cooldown=parms["cooldown"],
                                                                factor=parms["factor"],
                                                                min_lr=parms["min_lr"]) #
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                      schedulers=[scheduler_warmup, scheduler_main],
                                                      milestones=[parms["warmup_iters"]])

    print("number of parameters:", sum(p.numel() for p in model.parameters()))

    print()
    print("######################")
    print("# Training the model #")
    print("######################")
    #with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
    model, optimizer, scheduler, plotter = train_asymptotic_model(
        parms,
        model,
        optimizer,
        scheduler,
        plotter,
        dataset_asymptotic_train_list,
        dataset_asymptotic_test_list,
        dataset_stable_train_list,
        dataset_stable_test_list
    )
