The code in this repository was for HPC deployment as part of research using ensembles of neural networks as the non-linearity 
in a NARX approach to time series prediction for additive manufacturing.

There are two main code files: 

(a) testMLPtrain.py that trains multilayer perceptrons in a standard open loop configuration.
    The input data consists of laser power, head velocities and current meltpool depth.
    The output is the next meltpool depth in the sequence to create a one-step ahead non-linear predictor.

(b) testMLP.py tests the networks when configured as part of the recurrence in a NARX approach. The network is given
    an initial condition (velocity and laser power), makes a prediction of the next output, feeds predicted 
    output back to the input, and repeats (either for the same process parameters or for changing parameters. 
    e.g. changing laser power) generating a whole simulated sequence.

The networks are trained for a number of representative manufacturing process settings for a triangular process path.


 




