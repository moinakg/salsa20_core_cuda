salsa20_core_cuda
=================

Port of Salsa20 core crypto function to CUDA.

Running the code
================
You need to have the Nvidia GPU Computing SDK in order to try this code.
Lets assume that the SDK example codes are here:
~/NVIDIA_GPU_Computing_SDK/C/src/

Copy this entire directory into the above src dir so you get:
~/NVIDIA_GPU_Computing_SDK/C/src/salsa20_core_cuda

Now into this directory and run make. It should create a binary called
vecCrypt in the following location:
/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/

You can now run that binary as you would run any other sample program.

Tweaking the code
=================
Some of the constants in the code are suited to CUDA Capability 1.x devices
and can be tweaked for CUDA Capoability 2.x or later for better performance.

The constant THREADS_PER_BLOCK is dependent on the amount of shared memory
in the device. For devices with 48KB shared mem (2.x and later) this can be
increased to 340. Each thread uses 144 bytes of shared memory.

You can also experiment with BLOCKS_PER_CHUNK to see if a value other than 4
gives better results.

