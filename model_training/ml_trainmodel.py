import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_maxentropy import *

def train_asymptotic_model(parms,
                           model,
                           optimizer,
                           scheduler,
                           plotter,
                           dataset_size,
                           F4i_train,
                           F4f_train,
                           F4i_test,
                           F4f_test,
                           F4_NSM_stable_train,
                           F4_NSM_stable_test):
    print("Training dataset size:",dataset_size)

    # create a new plotter object of larger size if epochs is larger than the plotter object
    p = Plotter(parms["epochs"],["knownData","unphysical","0ff","1f","finalstable","randomstable","NSM_stable"])
    p.fill_from_plotter(plotter)

    #=====================================================#
    # Load training data into data loader for minibatches #
    #=====================================================#
    if dataset_size == -1:
        dataset_size = F4i_train.shape[0]
    assert(dataset_size <= F4i_train.shape[0])
    F4i_train = F4i_train[:dataset_size]
    F4f_train = F4f_train[:dataset_size]
    dataset = torch.utils.data.TensorDataset(F4i_train, F4f_train)

    #===============#
    # training loop #
    #===============#
    # generate randomized data and evaluate the test error
    F4i_unphysical_test = generate_random_F4(parms["n_generate"],
                                             parms["NF"],
                                             parms["device"],
                                             max_fluxfac=parms["generate_max_fluxfac"]) # 
    F4_0ff_stable_train = generate_stable_F4_zerofluxfac(parms["n_generate"], parms["NF"], parms["device"])
    F4_0ff_stable_test  = generate_stable_F4_zerofluxfac(parms["n_generate"], parms["NF"], parms["device"])
    F4_1f_stable_train  = generate_stable_F4_oneflavor(  parms["n_generate"], parms["NF"], parms["device"])
    F4_1f_stable_test   = generate_stable_F4_oneflavor(  parms["n_generate"], parms["NF"], parms["device"])

    # set up datasets of stable distributions based on the max entropy stability condition
    F4i_random = generate_random_F4(parms["n_generate"],
                                    parms["NF"],
                                    'cpu',
                                    zero_weight=parms["ME_stability_zero_weight"],
                                    max_fluxfac=parms["generate_max_fluxfac"])
    unstable_random = has_crossing(F4i_random.detach().numpy(),
                                   parms["NF"],
                                   parms["ME_stability_n_equatorial"]).squeeze()
    print("random Stable:",np.sum(unstable_random==False))
    print("random Unstable:",np.sum(unstable_random==True))
    F4_random_stable = F4i_random[unstable_random==False]
    F4_random_stable = augment_permutation(F4_random_stable)
    F4_random_stable = F4_random_stable.float().to(parms["device"])

    # split into a test and train set
    ntotal = F4_random_stable.shape[0]
    F4_random_stable_train = F4_random_stable[ntotal//2:]
    F4_random_stable_test = F4_random_stable[:ntotal//2]

    epochs_already_done = len(plotter.data["knownData"].train_loss)
    for t in range(epochs_already_done, parms["epochs"]):

        # generate more unphysical test data
        F4i_unphysical_train = generate_random_F4(parms["n_generate"]*10,
                                                  parms["NF"],
                                                  parms["device"],
                                                  max_fluxfac=parms["generate_max_fluxfac"])

        # zero the gradients
        optimizer.optimizer.zero_grad()

        def contribute_loss(F4pred_train, F4f_train, F4pred_test, F4f_test, key, loss_fn):
            this_loss = loss_fn(F4pred_train, F4f_train)
            p.data[key].train_loss[t] = this_loss.item()
            p.data[key].train_err[t]  = max_error(F4pred_train, F4f_train)
            
            p.data[key].test_loss[t] = loss_fn(F4pred_test, F4f_test)
            p.data[key].test_err[t]  = max_error(F4pred_test, F4f_test)

            return this_loss

        # train on making sure the model prediction is correct
        F4pred_test  = model.predict_F4(F4i_test ,"eval")
        F4pred_train = model.predict_F4(F4i_train,"train")
        loss = contribute_loss(F4pred_train, F4f_train, F4pred_test, F4f_test, "knownData", comparison_loss_fn)

        # final stable #
        F4pred_test = model.predict_F4(F4f_test,"eval")
        F4pred_train = model.predict_F4(F4f_train,"train")
        loss_final_stable = contribute_loss(F4pred_train, F4f_train, F4pred_test, F4f_test, "finalstable", comparison_loss_fn)
        if parms["do_augment_final_stable"]:
            loss += loss_final_stable

        # 1 flavor stable
        F4pred_test = model.predict_F4(F4_1f_stable_test,"eval")
        F4pred_train = model.predict_F4(F4_1f_stable_train,"train")
        loss_1f_stable = contribute_loss(F4pred_train, F4_1f_stable_train, F4pred_test, F4_1f_stable_test, "1f", comparison_loss_fn)
        if parms["do_augment_1f"]:
            loss += loss_1f_stable

        # 0 flux factor stable
        F4pred_test = model.predict_F4(F4_0ff_stable_test,"eval")
        F4pred_train = model.predict_F4(F4_0ff_stable_train,"train")
        loss_0ff_stable = contribute_loss(F4pred_train, F4_0ff_stable_train, F4pred_test, F4_0ff_stable_test, "0ff", comparison_loss_fn)
        if parms["do_augment_0ff"]:
            loss += loss_0ff_stable

        # random stable
        F4pred_test = model.predict_F4(F4_random_stable_test,"eval")
        F4pred_train = model.predict_F4(F4_random_stable_train,"train")
        loss_random_stable = contribute_loss(F4pred_train, F4_random_stable_train, F4pred_test, F4_random_stable_test, "randomstable", comparison_loss_fn)
        if parms["do_augment_random_stable"]:
            loss += loss_random_stable
            
        # NSM_stable
        F4pred_test = model.predict_F4(F4_NSM_stable_test,"eval")
        F4pred_train = model.predict_F4(F4_NSM_stable_train,"train")
        loss_NSM_stable = contribute_loss(F4pred_train, F4_NSM_stable_train, F4pred_test, F4_NSM_stable_test, "NSM_stable", comparison_loss_fn)
        if parms["do_augment_NSM_stable"]:
            loss += loss_NSM_stable

        # unphysical
        F4pred_test = model.predict_F4(F4i_unphysical_test,"test")
        F4pred_train = model.predict_F4(F4i_unphysical_train,"train")
        loss_unphysical = contribute_loss(F4pred_train, None, F4pred_test, None, "unphysical", unphysical_loss_fn) * 100
        if parms["do_unphysical_check"]:
            loss += loss_unphysical

        # have the optimizer take a step
        scheduler.step(loss.item())
        loss.backward()
        optimizer.optimizer.step()

        # report max error
        if((t+1)%parms["print_every"]==0):
            print(f"Epoch {t+1}")
            print("lr =",scheduler._last_lr)
            for key in p.data.keys():
                print(key, np.sqrt(p.data[key].train_loss[t]),  np.sqrt(p.data[key].test_loss[t]))
            
            print("", flush=True)

    return model, optimizer, scheduler, p
