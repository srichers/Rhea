import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *

def train_model(NF,
                do_fdotu,
                nhidden,
                width,
                dropout_probability,
                do_batchnorm,
                activation,
                op,
                weight_decay,
                learning_rate,
                epochs,
                batch_size,
                n_generate,
                dataset_size,
                print_every,
                device,
                conserve_lepton_number,
                do_augment_final_stable,
                do_NSM_stable,
                do_unphysical_check,
                do_trivial_stable,
                comparison_loss_fn,
                unphysical_loss_fn,
                F4i_train,
                F4f_train,
                F4i_test,
                F4f_test,
                F4_NSM_train,
                F4_NSM_test):
    #=======================#
    # instantiate the model #
    #=======================#
    model = NeuralNetwork(NF,
                          do_fdotu,
                          nhidden,
                          width,
                          dropout_probability,
                          activation,
                          do_batchnorm).to(device)
    optimizer = Optimizer(model,
                          op,
                          weight_decay,
                          learning_rate,
                          device,
                          conserve_lepton_number=conserve_lepton_number)

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
    p = Plotter(epochs)
    for t in range(epochs):

        # generate randomized data and evaluate the test error
        p.knownData.test_loss[t],  p.knownData.test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  comparison_loss_fn)
        if do_augment_final_stable:
            p.knownData_FS.test_loss[t],  p.knownData_FS.test_err[t]  = optimizer.test(model, F4f_test,  F4f_test,  comparison_loss_fn)
        p.NSM.test_loss[t],  p.NSM.test_err[t]  = optimizer.test(model, F4_NSM_test,  F4_NSM_test,  comparison_loss_fn)
        if do_unphysical_check:
            F4i_unphysical = generate_random_F4(n_generate, NF, device)
            p.unphysical.test_loss[t],  p.unphysical.test_err[t]  = optimizer.test(model, F4i_unphysical, None, unphysical_loss_fn)
        if do_trivial_stable:
            F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
            p.zerofluxfac.test_loss[t],  p.zerofluxfac.test_err[t]  = optimizer.test(model, F4i_0ff, F4i_0ff, comparison_loss_fn)
            F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
            p.oneflavor.test_loss[t],  p.oneflavor.test_err[t]  = optimizer.test(model, F4i_1f, F4i_1f, comparison_loss_fn)


        # load in a batch of data from the dataset
        for F4i_batch, F4f_batch in dataloader:

            # zero the gradients
            optimizer.optimizer.zero_grad()

            # train on making sure the model prediction is correct
            loss = optimizer.train(model, F4i_batch, F4f_batch, comparison_loss_fn)
            loss.backward()

            if do_augment_final_stable:
                loss = optimizer.train(model, F4f_batch, F4f_batch, comparison_loss_fn)
                loss.backward()

            if do_NSM_stable:
                loss = optimizer.train(model, F4_NSM_train, F4_NSM_train, comparison_loss_fn)
                loss.backward()

            # train on making sure the model prediction is physical
            if do_unphysical_check:
                loss = optimizer.train(model, F4i_unphysical, None, unphysical_loss_fn)
                loss.backward()

            # train on making sure known stable distributions dont change
            if do_trivial_stable:
                loss = optimizer.train(model, F4i_0ff, F4i_0ff, comparison_loss_fn)
                loss.backward()

                loss = optimizer.train(model, F4i_1f, F4i_1f, comparison_loss_fn)
                loss.backward()

            # take a step with the optimizer    
            optimizer.optimizer.step()

        # Evaluate training errors
        p.knownData.train_loss[t], p.knownData.train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn)
        if do_augment_final_stable:
            p.knownData_FS.train_loss[t], p.knownData_FS.train_err[t] = optimizer.test(model, F4f_train, F4f_train, comparison_loss_fn)
        if do_NSM_stable:
            p.NSM.train_loss[t], p.NSM.train_err[t] = optimizer.test(model, F4_NSM_train, F4_NSM_train, comparison_loss_fn)    

        # report max error
        if((t+1)%print_every==0):
            print(f"Epoch {t+1}")
            print("Train loss:",      p.knownData.train_loss[t])
            print("Test loss:",       p.knownData.test_loss[t])
            if do_augment_final_stable:
                print("Test FS loss:",       p.knownData_FS.test_loss[t])
            print("Test NSM loss:",       p.NSM.test_loss[t])
            if do_unphysical_check:
                print("Test unphysical loss:",       p.unphysical.test_loss[t])
            if do_trivial_stable:
                print("Test zerofluxfac loss:",       p.zerofluxfac.test_loss[t])
                print("Test oneflavor loss:",       p.oneflavor.test_loss[t])
            
            print()

    return model, p
