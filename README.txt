The code in this repository was for HPC deployment as part of research using ensembles of neural networks as the non-linearity in a NARX approach to time series prediction for additive manufacturing.

There are two main code files: 

(a) testMLPtrain that trains multilayer perceptrons in a standard open loop configuration
(b) testMLP tests the networks when configured as part of the recurrence in a NARX approach. The network is given
    an initial condition, makes a prediction of the next output, feeds predicted output back to the input.

The networks are trained for a number of representative manufacturing process settings for a triangle process path.

 