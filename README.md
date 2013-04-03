salsa20_core_cuda
=================

Port of Salsa20 core crypto function to CUDA.

Running the code
================
You can simply run make within the directory to build the code if the
CUDA 5.0 toolkit is installed in the default location. If CUDA is installed
in some other place or you are using an older CUDA toolkit then you can run
make like this:

make CUDA_PATH=/path/to/cuda/toolkit

The following executable binaries will be generated:

vecCrypt - Standard single-stream GPU based Salsa20 encryption code.
vecCrypt_strm - CUDA Streams based overlapped copy and execute version using
                2 streams.

Tweaking the code
=================
Some of the constants in the code are suited to CUDA Capability 1.x devices
and can be tweaked for CUDA Capability 2.x or later for better performance.

The constant THREADS_PER_BLOCK is dependent on the amount of registers and
shared memory in the device. It can be increased for devices with CUDA
Capability 2.x and later.

