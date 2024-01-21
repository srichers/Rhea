import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_maxentropy import *

def train_asymptotic_model(model,
                optimizer,
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
                comparison_loss_fn,
                unphysical_loss_fn,
                F4i_train,
                F4f_train,
                F4i_test,
                F4f_test):
    print("Training dataset size:",dataset_size)

    # create a new plotter object of larger size if epochs is larger than the plotter object
    p = Plotter(epochs,["knownData","unphysical","knownData_corrected","unphysical_corrected"])
    p.fill_from_plotter(plotter)

    #=====================================================#
    # Load training data into data loader for minibatches #
    #=====================================================#
    if dataset_size == -1:
        dataset_size = F4i_train.shape[0]
    if batch_size == -1:
        batch_size = dataset_size
    assert(dataset_size <= F4i_train.shape[0])
    assert(batch_size <= dataset_size)
    F4i_train = F4i_train[:dataset_size]
    F4f_train = F4f_train[:dataset_size]
    dataset = torch.utils.data.TensorDataset(F4i_train, F4f_train)
    batch_size = max(batch_size, len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("batchsize=",batch_size)

    #===============#
    # training loop #
    #===============#
    epochs_already_done = len(plotter.data["knownData"].train_loss)
    for t in range(epochs_already_done, epochs):

        # generate randomized data and evaluate the test error
        F4i_unphysical = generate_random_F4(n_generate, NF, device, max_fluxfac=generate_max_fluxfac)

        # log the test error
        p.data["knownData"].test_loss[t],  p.data["knownData"].test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  comparison_loss_fn, False, False)
        p.data["unphysical"].test_loss[t],  p.data["unphysical"].test_err[t]  = optimizer.test(model, F4i_unphysical, None, unphysical_loss_fn, False, False)
        p.data["knownData_corrected"].test_loss[t],  p.data["knownData_corrected"].test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  comparison_loss_fn, True, True)
        p.data["unphysical_corrected"].test_loss[t],  p.data["unphysical_corrected"].test_err[t]  = optimizer.test(model, F4i_unphysical, None, unphysical_loss_fn, True,True)

        # load in a batch of data from the dataset
        with torch.autograd.detect_anomaly():
            for F4i_batch, F4f_batch in dataloader:

                # zero the gradients
                optimizer.optimizer.zero_grad()

                # train on making sure the model prediction is correct
                loss = optimizer.train(model, F4i_batch, F4f_batch, comparison_loss_fn, False,False)
                assert(loss==loss)
                loss.backward()

                # train on making sure the model prediction is physical
                if do_unphysical_check:
                    loss = optimizer.train(model, F4i_unphysical, None, unphysical_loss_fn, False,False)
                    assert(loss==loss)
                    loss.backward()

                optimizer.optimizer.step()

        # Evaluate training errors
        p.data["knownData"].train_loss[t], p.data["knownData"].train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn,False,False)
        p.data["knownData_corrected"].train_loss[t], p.data["knownData_corrected"].train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn,True,True)

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            for key in p.data.keys():
                print(key, p.data[key].train_loss[t],  p.data[key].test_loss[t])
            
            print()

    return model, optimizer, p


def train_stability_model(model,
                optimizer,
                plotter,
                NF,
                epochs,
                n_generate,
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
    F4i_random = generate_random_F4(n_generate, NF, 'cpu', zero_weight=zero_weight)
    unstable_random = has_crossing(F4i_random.detach().numpy(), NF, n_equatorial)
    print("Random Stable:",np.sum(unstable_random==False))
    print("Random Unstable:",np.sum(unstable_random==True))

    F4i_heavy = generate_random_F4(n_generate, NF, 'cpu', zero_weight=zero_weight)
    F4i_heavy[:,:,:,1:] = F4i_heavy[:,:,0,1][:,:,None,None]
    F4i_heavy = augment_permutation(F4i_heavy)
    unstable_heavy = has_crossing(F4i_heavy.detach().numpy(), NF, n_equatorial)
    print("Heavy Stable:",np.sum(unstable_random==False))
    print("Heavy Unstable:",np.sum(unstable_random==True))

    F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
    unstable_0ff = torch.zeros((n_generate,1), device=device)

    F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
    unstable_1f = torch.zeros((F4i_1f.shape[0],1), device=device)

    # move data to device
    F4i_random = torch.tensor(F4i_random).float().to(device)
    unstable_random = torch.tensor(unstable_random).to(device)
    F4i_heavy = torch.tensor(F4i_heavy).float().to(device)
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

        loss = optimizer.train(model, F4i_heavy, unstable_heavy, loss_function)
        loss.backward()

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

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            for key in p.data.keys():
                print(key, p.data[key].train_loss[t],  p.data[key].test_loss[t])            
            print()

    return model, optimizer, p
