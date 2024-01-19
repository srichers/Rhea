import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *

def train_asymptotic_model(model,
                optimizer,
                plotter,
                NF,
                epochs,
                batch_size,
                n_generate,
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
    p = Plotter(epochs)
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
    epochs_already_done = len(plotter.knownData.train_loss)
    for t in range(epochs_already_done, epochs):

        # generate randomized data and evaluate the test error
        F4i_unphysical = generate_random_F4(n_generate, NF, device)
        F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
        F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)

        # log the test error
        p.knownData.test_loss[t],  p.knownData.test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  comparison_loss_fn)
        p.unphysical.test_loss[t],  p.unphysical.test_err[t]  = optimizer.test(model, F4i_unphysical, None, unphysical_loss_fn)

        # load in a batch of data from the dataset
        for F4i_batch, F4f_batch in dataloader:

            # zero the gradients
            optimizer.optimizer.zero_grad()

            # train on making sure the model prediction is correct
            loss = optimizer.train(model, F4i_batch, F4f_batch, comparison_loss_fn)
            loss.backward()

            # train on making sure the model prediction is physical
            if do_unphysical_check:
                loss = optimizer.train(model, F4i_unphysical, None, unphysical_loss_fn)
                loss.backward()

            # take a step with the optimizer    
            optimizer.optimizer.step()

        # Evaluate training errors
        p.knownData.train_loss[t], p.knownData.train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn)

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            print("Train loss:",      p.knownData.train_loss[t])
            print("Test loss:",       p.knownData.test_loss[t])
            print("Test unphysical loss:",       p.unphysical.test_loss[t])
            
            print()

    return model, optimizer, p
