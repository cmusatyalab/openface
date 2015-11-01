# How long does processing a face take?
The processing time depends on the size of your image for
face detection and alignment.
These only run on the CPU and take from 100-200ms to over
a second.
The neural network uses a fixed-size input and has
a more consistent runtime.
Averaging over 500 forward passes of random input, the latency is
77.47 ms &plusmn; 50.69 ms on our 3.70 GHz CPU and
21.13 ms &plusmn; 6.15 ms on our Tesla K40 GPU,
obtained with
[util/profile-network.lua](util/profile-network.lua)
