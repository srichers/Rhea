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
    F4i_0ff_train = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    F4i_0ff_test = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    F4i_1f_train = generate_stable_F4_oneflavor(n_generate, NF, device)
    F4i_1f_test = generate_stable_F4_oneflavor(n_generate, NF, device)

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

        # log the test error
        p.data["knownData"].test_loss[t],  p.data["knownData"].test_err[t]  = optimizer.test(model, F4i_test,  F4f_test, comparison_loss_fn)
        p.data["unphysical"].test_loss[t],  p.data["unphysical"].test_err[t]  = optimizer.test(model, F4i_unphysical_test, None, unphysical_loss_fn)
        p.data["0ff"].test_loss[t],  p.data["0ff"].test_err[t]  = optimizer.test(model, F4i_0ff_test, F4i_0ff_test, comparison_loss_fn)
        p.data["1f"].test_loss[t],  p.data["1f"].test_err[t]  = optimizer.test(model, F4i_1f_test, F4i_1f_test, comparison_loss_fn)
        p.data["finalstable"].test_loss[t],  p.data["finalstable"].test_err[t]  = optimizer.test(model, F4f_train, F4f_train, comparison_loss_fn)
        p.data["randomstable"].test_loss[t],  p.data["randomstable"].test_err[t]  = optimizer.test(model, F4_random_stable_test, F4_random_stable_test, comparison_loss_fn)
        p.data["NSM_stable"].test_loss[t],  p.data["NSM_stable"].test_err[t]  = optimizer.test(model, F4_NSM_stable_test, F4_NSM_stable_test, comparison_loss_fn)

        # zero the gradients
        optimizer.optimizer.zero_grad()

        # set the model to training mode
        model.train()

        # train on making sure the model prediction is correct
        F4f_pred = model.predict_F4(F4i_train)
        loss = comparison_loss_fn(F4f_pred, F4f_train)
        
        if do_augment_final_stable:
            loss = loss + optimizer.train(model, F4f_train, F4f_train, comparison_loss_fn)
            
        if do_augment_1f:
            loss = loss + optimizer.train(model, F4i_1f_train, F4i_1f_train, comparison_loss_fn)
            
        if do_augment_0ff:
            loss = loss + optimizer.train(model, F4i_0ff_train, F4i_0ff_train, comparison_loss_fn)

        if do_augment_random_stable:
            loss = loss + optimizer.train(model, F4_random_stable_train, F4_random_stable_train, comparison_loss_fn)

        if do_augment_NSM_stable:
            loss = loss + optimizer.train(model, F4_NSM_stable_train, F4_NSM_stable_train, comparison_loss_fn)
            
        # train on making sure the model prediction is physical
        if do_unphysical_check:
            loss = loss + optimizer.train(model, F4i_unphysical_train, None, unphysical_loss_fn) * 100
            
        loss.backward()
        optimizer.optimizer.step()

        # Evaluate training errors
        p.data["knownData"].train_loss[t], p.data["knownData"].train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn)
        p.data["unphysical"].train_loss[t], p.data["unphysical"].train_err[t] = optimizer.test(model, F4i_unphysical_train, None, unphysical_loss_fn)
        p.data["0ff"].train_loss[t], p.data["0ff"].train_err[t] = optimizer.test(model, F4i_0ff_train, F4i_0ff_train, comparison_loss_fn)
        p.data["1f"].train_loss[t], p.data["1f"].train_err[t] = optimizer.test(model, F4i_1f_train, F4i_1f_train, comparison_loss_fn)
        p.data["finalstable"].train_loss[t], p.data["finalstable"].train_err[t] = optimizer.test(model, F4f_train, F4f_train, comparison_loss_fn)
        p.data["randomstable"].train_loss[t], p.data["randomstable"].train_err[t] = optimizer.test(model, F4_random_stable_train, F4_random_stable_train, comparison_loss_fn)
        p.data["NSM_stable"].train_loss[t], p.data["NSM_stable"].train_err[t] = optimizer.test(model, F4_NSM_stable_train, F4_NSM_stable_train, comparison_loss_fn)

        # update the learning rate
        netloss = p.data["knownData"].train_loss[t] + p.data["unphysical"].train_loss[t]*100 + p.data["0ff"].train_loss[t]*100 + p.data["1f"].train_loss[t]*100 + p.data["finalstable"].train_loss[t]
        scheduler.step(netloss)

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            print("lr =",scheduler._last_lr)
            for key in p.data.keys():
                print(key, np.sqrt(p.data[key].train_loss[t]),  np.sqrt(p.data[key].test_loss[t]))
            
            print("", flush=True)

    return model, optimizer, scheduler, p
