salsa20_core_cuda
=================

Port of Salsa20 core crypto function to CUDA.

Running the code
================
You can simply run "make" within the directory to build the code if the
CUDA 5.0 toolkit is installed in the default location. If CUDA is installed
in some other place or you are using an older CUDA toolkit then you can run
make like this:

make CUDA_PATH=/path/to/cuda/toolkit

The following executable binaries will be generated:

vecCrypt - Standard single-stream GPU based Salsa20 encryption code.

vecCrypt_strm - CUDA Streams based overlapped copy and execute version.
           It uses 16 streams by default. This is typically faster than
           the previous version.

vecCrypt_strm_cpuxor - CUDA Streams based overlapped copy and execute
           version with final CTR mode XOR done on the CPU. It involves
           PCI data transfer of only the keystream back to the host.
           Each CUDA stream gets a corresponding host thread that does
           the XOR once the stream finishes.
           The XOR is done using optimized SSE2 intrinsics. All buffers
           are aligned at 16-byte boundary so normal SSE loads are used.
           This version is by far the fastest of the three.

Tweaking the code
=================
Some of the constants in the file "common.h" are suited to CUDA Capability
1.x devices and can be tweaked for CUDA Capability 2.x or later for better
performance.

The constant THREADS_PER_BLOCK is dependent on the amount of registers and
shared memory in the device. It can be increased for devices with CUDA
Capability 2.x and later.

The constant NUM_STREAMS determines the number of CUDA streams to be used.
This can be tweaked to check results with different number of streams. The
test buffer (approx 244MB) is split up into these overlapped streams to
hide PCI transfer latency.
