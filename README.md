# Rhea
Rhea is a collection of scripts to train a simple neural network to predict the results of neutrino flavor instabilities by reproducing the results of detailed simulations.

### model_training
The model_training folder contains all of the scripts we use to collect data from Emu simulations, format them into a training dataset, and construct and optimize a neural network to reproduce the data. Some of the references are specific to the current workstation the code is being developed on, since the code is still in the exploratory phase.

### cpp_interface
The cpp_interface folder contains a header file with a class that loads and executes the ML model, with the goal of making it easy to implement in high-performance simulations that use moment-based neutrino transport.
