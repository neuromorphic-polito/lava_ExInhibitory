# 01-FP_visualization.ipynb
 - Loads the network in floating
 - Converts it to fixed point, using the library float2fixed
 - Runs the fixed network and check accuracy

To check visually that it behaves similarly to the snn version

# 02-FP_net_test.ipynb

 - Loads the floating point network
 - Runs on the validation dataset
 - Check the accuracy

To check that the accuracy is close to the one obtained with snn torch

# 03-FP_to_FIXED_conversion.ipynb

 - Declares the network (same for fixed or float)
 - create the converter from the float2fixed library
 - The library will run the floating version, and used the fixed version to understand the ranges for quantization.
 - save the fixed version of the network

Just to convert and save the fixed version of the network

# 04-Fixed_conversion_validation.ipynb

 - Declares the network (same for fixed or float)
 - create the converter from the float2fixed library
 - The library will run the floating version, and used the fixed version to understand the ranges for quantization.
 - Run the fixed network and check accuracy

To check the accuracy of the fixed network, in CPU

# 05-FIXED_visualization.ipynb

 - Loads the fixed point network
 - Run in a single sample
 - Generate plots

To check visually that it behaves similarly to the FP lava version

# 06-FIXED_net_test.ipynb

 - Loads the fixed point network
 - Runs on the validation dataset
 - Check the accuracy

To check that the accuracy is close to the one obtained with lava FP version

# 07-FP_FIXED_visualization.ipynb

 - Load FP version
 - Load Fixed version
 - Run for a single segment (6 channels, 40 timesteps)
 - Compare visually

All done in CPU
