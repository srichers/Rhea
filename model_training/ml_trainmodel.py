import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_maxentropy import *

def train_asymptotic_model(model,
                optimizer,
                scheduler,
                plotter,
                NF,
                epochs,
                batch_size,
                n_generate,
                generate_max_fluxfac,
                dataset_size,
                print_every,
                device,
                do_unphysical_check,
                do_augment_final_stable,
                do_augment_1f,
                do_augment_0ff,
                do_augment_random_stable,
                do_augment_NSM_stable,
                ME_stability_zero_weight,
                ME_stability_n_equatorial,
                comparison_loss_fn,
                unphysical_loss_fn,
                F4i_train,
                F4f_train,
                F4i_test,
                F4f_test,
                F4_NSM_stable_train,
                F4_NSM_stable_test):
    print("Training dataset size:",dataset_size)

    # create a new plotter object of larger size if epochs is larger than the plotter object
    p = Plotter(epochs,["knownData","unphysical","0ff","1f","finalstable","randomstable","NSM_stable"])
    p.fill_from_plotter(plotter)

    #=====================================================#
    # Load training data into data loader for minibatches #
    #=====================================================#
    if dataset_size == -1:
        dataset_size = F4i_train.shape[0]
    if batch_size == -1:
        batch_size = dataset_size
    assert(dataset_size <= F4i_train.shape[0])
    F4i_train = F4i_train[:dataset_size]
    F4f_train = F4f_train[:dataset_size]
    dataset = torch.utils.data.TensorDataset(F4i_train, F4f_train)
    batch_size = min(batch_size, len(dataset))
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("batchsize=",batch_size)

    #===============#
    # training loop #
    #===============#
    # generate randomized data and evaluate the test error
    F4i_unphysical_test = generate_random_F4(n_generate, NF, device, max_fluxfac=generate_max_fluxfac)
    F4_0ff_stable_train = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    F4_0ff_stable_test  = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    F4_1f_stable_train  = generate_stable_F4_oneflavor(  n_generate, NF, device)
    F4_1f_stable_test   = generate_stable_F4_oneflavor(  n_generate, NF, device)

    # set up datasets of stable distributions based on the max entropy stability condition
    F4i_random = generate_random_F4(n_generate, NF, 'cpu', zero_weight=ME_stability_zero_weight, max_fluxfac=generate_max_fluxfac)
    unstable_random = has_crossing(F4i_random.detach().numpy(), NF, ME_stability_n_equatorial).squeeze()
    print("random Stable:",np.sum(unstable_random==False))
    print("random Unstable:",np.sum(unstable_random==True))
    F4_random_stable = F4i_random[unstable_random==False]
    F4_random_stable = augment_permutation(F4_random_stable)
    F4_random_stable = F4_random_stable.float().to(device)

    # split into a test and train set
    ntotal = F4_random_stable.shape[0]
    F4_random_stable_train = F4_random_stable[ntotal//2:]
    F4_random_stable_test = F4_random_stable[:ntotal//2]

    epochs_already_done = len(plotter.data["knownData"].train_loss)
    for t in range(epochs_already_done, epochs):

        # generate more unphysical test data
        F4i_unphysical_train = generate_random_F4(n_generate*10, NF, device, max_fluxfac=generate_max_fluxfac)

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
        
        if do_augment_final_stable:
            F4pred_test = model.predict_F4(F4f_test,"eval")
            F4pred_train = model.predict_F4(F4f_train,"train")
            loss += contribute_loss(F4pred_train, F4f_train, F4pred_test, F4f_test, "finalstable", comparison_loss_fn)
            
        if do_augment_1f:
            F4pred_test = model.predict_F4(F4_1f_stable_test,"eval")
            F4pred_train = model.predict_F4(F4_1f_stable_train,"train")
            loss += contribute_loss(F4pred_train, F4_1f_stable_train, F4pred_test, F4_1f_stable_test, "1f", comparison_loss_fn)
            
        if do_augment_0ff:
            F4pred_test = model.predict_F4(F4_0ff_stable_test,"eval")
            F4pred_train = model.predict_F4(F4_0ff_stable_train,"train")
            loss += contribute_loss(F4pred_train, F4_0ff_stable_train, F4pred_test, F4_0ff_stable_test, "0ff", comparison_loss_fn)

        if do_augment_random_stable:
            F4pred_test = model.predict_F4(F4_random_stable_test,"eval")
            F4pred_train = model.predict_F4(F4_random_stable_train,"train")
            loss += contribute_loss(F4pred_train, F4_random_stable_train, F4pred_test, F4_random_stable_test, "randomstable", comparison_loss_fn)

        if do_augment_NSM_stable:
            F4pred_test = model.predict_F4(F4_NSM_stable_test,"eval")
            F4pred_train = model.predict_F4(F4_NSM_stable_train,"train")
            loss += contribute_loss(F4pred_train, F4_NSM_stable_train, F4pred_test, F4_NSM_stable_test, "NSM_stable", comparison_loss_fn)
            
        if do_unphysical_check:
            F4pred_test = model.predict_F4(F4i_unphysical_test,"test")
            F4pred_train = model.predict_F4(F4i_unphysical_train,"train")
            loss += contribute_loss(F4pred_train, None, F4pred_test, None, "NSM_stable", unphysical_loss_fn) * 100

        # have the optimizer take a step
        scheduler.step(loss.item())
        loss.backward()
        optimizer.optimizer.step()

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            print("lr =",scheduler._last_lr)
            for key in p.data.keys():
                print(key, np.sqrt(p.data[key].train_loss[t]),  np.sqrt(p.data[key].test_loss[t]))
            
            print("", flush=True)

    return model, optimizer, scheduler, p
