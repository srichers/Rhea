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

        # load in a batch of data from the dataset
        #if True: #with torch.autograd.detect_anomaly():
        #    for F4i_batch, F4f_batch in dataloader:
        F4i_batch = F4i_train
        F4f_batch = F4f_train

        # zero the gradients
        optimizer.optimizer.zero_grad()

        # train on making sure the model prediction is correct
        loss = optimizer.train(model, F4i_batch, F4f_batch, comparison_loss_fn)
        loss.backward()
        
        if do_augment_final_stable:
            loss = optimizer.train(model, F4f_batch, F4f_batch, comparison_loss_fn)
            loss.backward()
            
        if do_augment_1f:
            loss = optimizer.train(model, F4i_1f_train, F4i_1f_train, comparison_loss_fn)
            loss.backward()
            
        if do_augment_0ff:
            loss = optimizer.train(model, F4i_0ff_train, F4i_0ff_train, comparison_loss_fn)
            loss.backward()

        if do_augment_random_stable:
            loss = optimizer.train(model, F4_random_stable_train, F4_random_stable_train, comparison_loss_fn)
            loss.backward()

        if do_augment_NSM_stable:
            loss = optimizer.train(model, F4_NSM_stable_train, F4_NSM_stable_train, comparison_loss_fn)
            loss.backward()
            
        # train on making sure the model prediction is physical
        if do_unphysical_check:
            loss = optimizer.train(model, F4i_unphysical_train, None, unphysical_loss_fn) * 100
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
            
            print()

    return model, optimizer, scheduler, p


def train_stability_model(model,
                optimizer,
                scheduler,
                plotter,
                NF,
                epochs,
                n_generate,
                generate_max_fluxfac,
                print_every,
                device,
                n_equatorial,
                zero_weight,
                loss_function):
    
    print("Training dataset size:",n_generate)

    # create a new plotter object of larger size if epochs is larger than the plotter object
    p = Plotter(epochs,["random","heavy","0ff","1f"])
    p.fill_from_plotter(plotter)

    # set up training datasets
    F4i_random = generate_random_F4(n_generate, NF, 'cpu', zero_weight=zero_weight, max_fluxfac=generate_max_fluxfac)
    unstable_random = has_crossing(F4i_random.detach().numpy(), NF, n_equatorial)
    print("Random Stable:",np.sum(unstable_random==False))
    print("Random Unstable:",np.sum(unstable_random==True))

    F4i_heavy = generate_random_F4(n_generate, NF, 'cpu', zero_weight=zero_weight, max_fluxfac=generate_max_fluxfac)
    F4i_heavy[:,:,:,1:] = torch.mean(F4i_heavy[:,:,:,1:], dim=3)[:,:,:,None]
    F4i_heavy = augment_permutation(F4i_heavy)
    unstable_heavy = has_crossing(F4i_heavy.detach().numpy(), NF, n_equatorial)
    print("Heavy Stable:",np.sum(unstable_random==False))
    print("Heavy Unstable:",np.sum(unstable_random==True))

    F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    unstable_0ff = torch.zeros((n_generate,1), device=device)

    F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
    unstable_1f = torch.zeros((F4i_1f.shape[0],1), device=device)

    # move data to device
    F4i_random = F4i_random.float().to(device)
    unstable_random = torch.tensor(unstable_random).to(device)
    F4i_heavy = F4i_heavy.float().to(device)
    unstable_heavy = torch.tensor(unstable_heavy).to(device)

    #===============#
    # training loop #
    #===============#
    epochs_already_done = len(plotter.data["random"].train_loss)
    for t in range(epochs_already_done, epochs):

        # log the test error
        p.data["random"].test_loss[t],  p.data["random"].test_err[t]  = optimizer.test(model, F4i_random,  unstable_random,  loss_function)
        p.data["heavy"].test_loss[t],  p.data["heavy"].test_err[t]  = optimizer.test(model, F4i_heavy, unstable_heavy, loss_function)
        p.data["0ff"].test_loss[t],  p.data["0ff"].test_err[t]  = optimizer.test(model, F4i_0ff, unstable_0ff, loss_function)
        p.data["1f"].test_loss[t],  p.data["1f"].test_err[t]  = optimizer.test(model, F4i_1f, unstable_1f, loss_function)

        # zero the gradients
        optimizer.optimizer.zero_grad()

        # train the model
        loss = optimizer.train(model, F4i_random, unstable_random, loss_function)
        loss.backward()

        #loss = optimizer.train(model, F4i_heavy, unstable_heavy, loss_function)
        #loss.backward()

        loss = optimizer.train(model, F4i_0ff, unstable_0ff, loss_function)
        loss.backward()

        loss = optimizer.train(model, F4i_1f, unstable_1f, loss_function)
        loss.backward()

        # take a step with the optimizer    
        optimizer.optimizer.step()

        # Evaluate training errors
        p.data["random"].train_loss[t], p.data["random"].train_err[t] = optimizer.test(model, F4i_random, unstable_random, loss_function)
        p.data["heavy"].train_loss[t], p.data["heavy"].train_err[t] = optimizer.test(model, F4i_heavy, unstable_heavy, loss_function)
        p.data["0ff"].train_loss[t], p.data["0ff"].train_err[t] = optimizer.test(model, F4i_0ff, unstable_0ff, loss_function)
        p.data["1f"].train_loss[t], p.data["1f"].train_err[t] = optimizer.test(model, F4i_1f, unstable_1f, loss_function)

        # update the learning rate
        netloss = p.data["random"].train_loss[t] + p.data["heavy"].train_loss[t] + p.data["0ff"].train_loss[t] + p.data["1f"].train_loss[t]
        scheduler.step()#netloss)

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            print("lr =",scheduler._last_lr)
            for key in p.data.keys():
                print(key, p.data[key].train_loss[t],  p.data[key].test_loss[t])            
            print()

    return model, optimizer, scheduler, p
