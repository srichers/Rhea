# Rhea
Rhea is a collection of scripts to train a simple neural network to predict the results of neutrino flavor instabilities by reproducing the results of detailed simulations.

### model_training
The model_training folder contains all of the scripts we use to collect data from Emu simulations, format them into a training dataset, and construct and optimize a neural network to reproduce the data. Some of the references are specific to the current workstation the code is being developed on, since the code is still in the exploratory phase.

#### Parameter Descriptions
The file <pre><code>ml_pytorch.py</code></pre> is the main script that creates a trained model. Below are a description of the parameters used:

[filenames]
* <pre><code>basedir = "/mnt/scratch/srichers/ML_FFI"</code></pre> The working directory. Input data is assumed to lie in a subdirectory called <pre><code>input_data</code></pre>.
* <pre><code>directory_list = ["manyflavor_twobeam", "manyflavor_twobeam_z"]</code></pre> Uses simulated data from each directory in the list.

[training]
* <pre><code>test_size = 0.1</code></pre> 10% of the data is used as the test set. The other 90% is used for training.
* <pre><code>epochs = 100</code></pre> The training algorithm will go through 50 epochs of training.
* <pre><code>batch_size = 100</code></pre> Within a single epoch, the algorithm will optimize on each batch of data separately. A smaller batch size is useful when the GPU does not have enough memory to fit the full dataset, or when it is beneficial to introduce some noise into the gradient descent to help avoid local minima. Which data go into which batch are shuffled each epoch. Will max out at the input data size, irrespective of input value.
* <pre><code>print_every = 1</code></pre> Print status after every 1 epoch

[data augmentation]
* <pre><code>do_augmentation_permutation = True</code></pre> Augment the data with flavor and charge permutations of the input/output data. For instance, in the fast flavor limit, we expect the same result for intersecting electron neutrino/antineutrino beams as intersecting muon neutrino/antineutrino beams.
* <pre><code>do_augment_final_stable = True</code></pre> The input data includes the initial and final states of many simulations. Setting this to true will augment the data with the same simulations, but setting both the starting and ending values to the final state of the simulation. This attempts to force the ML model to require that, once a distribution has undergone the FFI, it is no longer susceptible to flavor change.
* <pre><code>do_unphysical_check = True</code></pre> If set to True, generate a random set of moments of size <pre><code>batch_size</code></pre>. The ML model will predict results, and a penalty will be added to the loss for negative number densities or flux factors larger than 1.
* <pre><code>do_particlenumber_conservation_check = True</code></pre> If set to True, add to the loss when flavor-traced moments are not conserved. The ML model is built to conserve particle number by construction, so this should never have an impact unless something is broken.
* <pre><code>do_trivial_stable = True</code></pre> During each epoch, generate two sets of random sets of data known to be stable, each with size <pre><code>batch_size</code></pre>. The first has random number densities but all zero flux factors, while the second has random numbers, but only one flavor.

[neural network]
* <pre><code>conserve_lepton_number = True</code></pre> If set to True, before the ML model outputs a value, calculate the non-conservation of lepton number for all three flavors and symmetrically adjust the number densities so the end result is no change in lepton number.
* <pre><code>nhidden = 1</code></pre> Number of hidden layers. More layers allows for more detail, but increases the odds of overtraining and the time require to train.
* <pre><code>width = 32</code></pre> Width of the hidden layers. More width allows for more detail, but increases the odds of overtraining and the time required to train.
* <pre><code>dropout_probability = 0.5</code></pre> During training, probability of a node dropping out of the calculation. A higher dropout probability makes the neural network perform worse during training, but allows it to generalize better to unknown data.
* <pre><code>activation = nn.LeakyReLU</code></pre> Activation function for the nodes. LeakyReLU helps prevent nodes being stuck in the off state without any possibility for gradients to push it to another value.

[optimizer]
* <pre><code>op = torch.optim.Adam</code></pre> Choice of optimizer. Adam seems to be the goto for a first attempt.
* <pre><code>weight_decay = 1e-4</code></pre> Add a term to the loss function equal to the L2 norm of the weights, times this factor. This encourages the neural network to engage all nodes rather than having just a few large nodes dominating, which helps prevent overfitting and exploding gradients.
* <pre><code>learning_rate = 1e-3</code></pre> Rate at which to adjust weights when optimizing the neural network. Faster learning optimizes faster, but may be unstable. Adam adapts the learning rate for each parameter anyway, so this parameter serves as an upper limit to the learning rate.


### cpp_interface
The cpp_interface folder contains a header file with a class that loads and executes the ML model, with the goal of making it easy to implement in high-performance simulations that use moment-based neutrino transport.

