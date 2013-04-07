/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This software contains source code provided by NVIDIA Corporation.
 * 
 * GPU accelerated Salsa20 Vector crypto core function.
 *
 * This sample demonstrates an implementation of the core Salsa20 crypto function
 * in CTR mode accelerated using CUDA.
 */

// Includes
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>

// includes CUDA
#include <cuda_runtime.h>

#include "common.h"

extern "C" void xor_buffer_aligned(unsigned char *buf1, unsigned char *buf2, size_t nblks, int blksz);
extern "C" int crypto_stream_salsa20_amd64_xmm6_xor(unsigned char *c, unsigned char *m,
		unsigned long long mlen, unsigned char *n, unsigned char *k);

__constant__ unsigned char MY_ALIGN(sizeof (uint32_t)) key[XSALSA20_CRYPTO_KEYBYTES * THREADS_PER_BLOCK];
__constant__ unsigned char MY_ALIGN(sizeof (uint32_t)) sigma[16];
const unsigned char hsigma[17] = "expand 32-byte k";
unsigned char h_nonce[XSALSA20_CRYPTO_NONCEBYTES];
int pinned = 0;

__host__ __device__ static inline uint32_t
rotate(uint32_t u,int c)
{
  return (u << c) | (u >> (32 - c));
}

__host__ __device__ static inline uint32_t
load_littleendian(const unsigned char *x)
{
  return
      (uint32_t) (x[0]) \
  | (((uint32_t) (x[1])) << 8) \
  | (((uint32_t) (x[2])) << 16) \
  | (((uint32_t) (x[3])) << 24)
  ;
}

__host__ __device__ static inline void
store_littleendian(unsigned char *x, uint32_t u)
{
  x[0] = u; u >>= 8;
  x[1] = u; u >>= 8;
  x[2] = u; u >>= 8;
  x[3] = u;
}

__host__ static inline uint32_t
load_littleendian64(const unsigned char *x)
{
  return
      (uint64_t) (x[0]) \
  | (((uint64_t) (x[1])) << 8) \
  | (((uint64_t) (x[2])) << 16) \
  | (((uint64_t) (x[3])) << 24) \
  | (((uint64_t) (x[4])) << 32) \
  | (((uint64_t) (x[5])) << 40) \
  | (((uint64_t) (x[6])) << 48) \
  | (((uint64_t) (x[7])) << 56)
  ;
}


__host__ static int
crypto_core(
        unsigned char *out,
  const unsigned char *in,
  const unsigned char *k,
  const unsigned char *c
)
{
  uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
  uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;
  int i;

  j0 = x0 = load_littleendian(c + 0);
  j1 = x1 = load_littleendian(k + 0);
  j2 = x2 = load_littleendian(k + 4);
  j3 = x3 = load_littleendian(k + 8);
  j4 = x4 = load_littleendian(k + 12);
  j5 = x5 = load_littleendian(c + 4);
  j6 = x6 = load_littleendian(in + 0);
  j7 = x7 = load_littleendian(in + 4);
  j8 = x8 = load_littleendian(in + 8);
  j9 = x9 = load_littleendian(in + 12);
  j10 = x10 = load_littleendian(c + 8);
  j11 = x11 = load_littleendian(k + 16);
  j12 = x12 = load_littleendian(k + 20);
  j13 = x13 = load_littleendian(k + 24);
  j14 = x14 = load_littleendian(k + 28);
  j15 = x15 = load_littleendian(c + 12);

  for (i = ROUNDS;i > 0;i -= 2) {
     x4 ^= rotate( x0+x12, 7);
     x8 ^= rotate( x4+ x0, 9);
    x12 ^= rotate( x8+ x4,13);
     x0 ^= rotate(x12+ x8,18);
     x9 ^= rotate( x5+ x1, 7);
    x13 ^= rotate( x9+ x5, 9);
     x1 ^= rotate(x13+ x9,13);
     x5 ^= rotate( x1+x13,18);
    x14 ^= rotate(x10+ x6, 7);
     x2 ^= rotate(x14+x10, 9);
     x6 ^= rotate( x2+x14,13);
    x10 ^= rotate( x6+ x2,18);
     x3 ^= rotate(x15+x11, 7);
     x7 ^= rotate( x3+x15, 9);
    x11 ^= rotate( x7+ x3,13);
    x15 ^= rotate(x11+ x7,18);
     x1 ^= rotate( x0+ x3, 7);
     x2 ^= rotate( x1+ x0, 9);
     x3 ^= rotate( x2+ x1,13);
     x0 ^= rotate( x3+ x2,18);
     x6 ^= rotate( x5+ x4, 7);
     x7 ^= rotate( x6+ x5, 9);
     x4 ^= rotate( x7+ x6,13);
     x5 ^= rotate( x4+ x7,18);
    x11 ^= rotate(x10+ x9, 7);
     x8 ^= rotate(x11+x10, 9);
     x9 ^= rotate( x8+x11,13);
    x10 ^= rotate( x9+ x8,18);
    x12 ^= rotate(x15+x14, 7);
    x13 ^= rotate(x12+x15, 9);
    x14 ^= rotate(x13+x12,13);
    x15 ^= rotate(x14+x13,18);
  }

  x0 += j0;
  x1 += j1;
  x2 += j2;
  x3 += j3;
  x4 += j4;
  x5 += j5;
  x6 += j6;
  x7 += j7;
  x8 += j8;
  x9 += j9;
  x10 += j10;
  x11 += j11;
  x12 += j12;
  x13 += j13;
  x14 += j14;
  x15 += j15;

  store_littleendian(out + 0,x0);
  store_littleendian(out + 4,x1);
  store_littleendian(out + 8,x2);
  store_littleendian(out + 12,x3);
  store_littleendian(out + 16,x4);
  store_littleendian(out + 20,x5);
  store_littleendian(out + 24,x6);
  store_littleendian(out + 28,x7);
  store_littleendian(out + 32,x8);
  store_littleendian(out + 36,x9);
  store_littleendian(out + 40,x10);
  store_littleendian(out + 44,x11);
  store_littleendian(out + 48,x12);
  store_littleendian(out + 52,x13);
  store_littleendian(out + 56,x14);
  store_littleendian(out + 60,x15);

  return 0;
}

// Variables
unsigned char* h_A = NULL;
unsigned char* h_B = NULL;
unsigned char* d_A = NULL;
bool noprompt = false;

// Functions
void CleanupResources(void);
void Init(unsigned char*, size_t);
void ParseArguments(int, char**);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
	CleanupResources();
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
	CleanupResources();
        exit(-1);
    }
}

// end of CUDA Helper Functions


// Device code
__global__ void VecCrypt(unsigned char* A, unsigned int N, uint64_t nblocks, uint64_t p_nonce, int blks_per_chunk, uint64_t blkoff)
{
    uint64_t i = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    if (i < N) {
        int k, tot;
        uint32_t *mem;
        uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
        uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;
        uint64_t blockno, adj_blockno;

        blockno = i*blks_per_chunk;
        tot = (nblocks - blockno > blks_per_chunk) ? blks_per_chunk:(nblocks - blockno);

        for (k = 0; k < tot; k++) {
            j0 = x0 = load_littleendian(sigma + 0);
            j1 = x1 = load_littleendian(key + 0);
            j2 = x2 = load_littleendian(key + 4);
            j3 = x3 = load_littleendian(key + 8);
            j4 = x4 = load_littleendian(key + 12);
            j5 = x5 = load_littleendian(sigma + 4);

            adj_blockno = blockno + blkoff;
            j6 = x6 = p_nonce;
            j7 = x7 = p_nonce >> 32;
            j8 = x8 = adj_blockno;
            j9 = x9 = adj_blockno >> 32;

            j10 = x10 = load_littleendian(sigma + 8);
            j11 = x11 = load_littleendian(key + 16);
            j12 = x12 = load_littleendian(key + 20);
            j13 = x13 = load_littleendian(key + 24);
            j14 = x14 = load_littleendian(key + 28);
            j15 = x15 = load_littleendian(sigma + 12);

            for (i = ROUNDS;i > 0;i -= 2) {
                x4  ^= rotate( x0+x12, 7);
                x8  ^= rotate( x4+ x0, 9);
                x12 ^= rotate( x8+ x4,13);
                x0  ^= rotate(x12+ x8,18);
                x9  ^= rotate( x5+ x1, 7);
                x13 ^= rotate( x9+ x5, 9);
                x1  ^= rotate(x13+ x9,13);
                x5  ^= rotate( x1+x13,18);
                x14 ^= rotate(x10+ x6, 7);
                x2  ^= rotate(x14+x10, 9);
                x6  ^= rotate( x2+x14,13);
                x10 ^= rotate( x6+ x2,18);
                x3  ^= rotate(x15+x11, 7);
                x7  ^= rotate( x3+x15, 9);
                x11 ^= rotate( x7+ x3,13);
                x15 ^= rotate(x11+ x7,18);
                x1  ^= rotate( x0+ x3, 7);
                x2  ^= rotate( x1+ x0, 9);
                x3  ^= rotate( x2+ x1,13);
                x0  ^= rotate( x3+ x2,18);
                x6  ^= rotate( x5+ x4, 7);
                x7  ^= rotate( x6+ x5, 9);
                x4  ^= rotate( x7+ x6,13);
                x5  ^= rotate( x4+ x7,18);
                x11 ^= rotate(x10+ x9, 7);
                x8  ^= rotate(x11+x10, 9);
                x9  ^= rotate( x8+x11,13);
                x10 ^= rotate( x9+ x8,18);
                x12 ^= rotate(x15+x14, 7);
                x13 ^= rotate(x12+x15, 9);
                x14 ^= rotate(x13+x12,13);
                x15 ^= rotate(x14+x13,18);
            }

            x0 += j0;
            x1 += j1;
            x2 += j2;
            x3 += j3;
            x4 += j4;
            x5 += j5;
            x6 += j6;
            x7 += j7;
            x8 += j8;
            x9 += j9;
            x10 += j10;
            x11 += j11;
            x12 += j12;
            x13 += j13;
            x14 += j14;
            x15 += j15;

            mem = (unsigned int *)&A[blockno*XSALSA20_BLOCKSZ];
            *mem = x0;  mem++;
            *mem = x1;  mem++;
            *mem = x2;  mem++;
            *mem = x3;  mem++;
            *mem = x4;  mem++;
            *mem = x5;  mem++;
            *mem = x6;  mem++;
            *mem = x7;  mem++;
            *mem = x8;  mem++;
            *mem = x9;  mem++;
            *mem = x10;  mem++;
            *mem = x11;  mem++;
            *mem = x12;  mem++;
            *mem = x13;  mem++;
            *mem = x14;  mem++;
            *mem = x15;
            blockno++;
        }
    }
}

__host__ int
crypto_stream_salsa20_ref_xor(
  unsigned char *m,unsigned long long mlen,
  unsigned char *n,
  unsigned char *k
)
{
  unsigned char in[16];
  unsigned char block[64];
  int i;
  unsigned int u;
  unsigned int blk;

  if (!mlen) return 0;
  blk = 0;

  for (i = 0;i < 8;++i) in[i] = n[i];
  for (i = 8;i < 16;++i) in[i] = 0;

  while (mlen >= XSALSA20_BLOCKSZ) {
    crypto_core(block,in,k,hsigma);
    for (i = 0;i < XSALSA20_BLOCKSZ;++i) m[i] ^= block[i];

    u = 1;
    for (i = 8;i < 16;++i) {
      u += (unsigned int) in[i];
      in[i] = u;
      u >>= 8;
    }

    mlen -= XSALSA20_BLOCKSZ;
    m += XSALSA20_BLOCKSZ;
    blk++;
  }

  if (mlen) {
    crypto_core(block,in,k,hsigma);
    for (i = 0;i < mlen;++i) m[i] ^= block[i];
  }
  return 0;
}

__host__ double
get_wtime_millis(void)
{
    struct timespec ts;
    int rv;

    rv = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (rv == 0)
        return (ts.tv_sec * 1000 + ((double)ts.tv_nsec) / 1000000L);
    return (1);
}

#define	BYTES_TO_MB(x) ((x) / (1024 * 1024))

__host__ double
get_mb_s(uint64_t bytes, double diff)
{
	double bytes_sec;

	bytes_sec = ((double)bytes / diff) * 1000;
	return (BYTES_TO_MB(bytes_sec));
}

struct tdat {
    unsigned char *h_A1;
    unsigned char *h_B1;
    uint64_t nblks;
    int quit;
    cudaStream_t strm;
    pthread_t tid;
    sem_t start;
    sem_t done;
};

void *
proc_buf(void *dat)
{
    struct tdat *data = (struct tdat *)dat;

    while (1) {
        sem_wait(&(data->start));
        if (data->quit) break;

        checkCudaErrors( cudaStreamSynchronize(data->strm) );
        xor_buffer_aligned(data->h_A1, data->h_B1, data->nblks, XSALSA20_BLOCKSZ);
        sem_post(&(data->done));
    }

    return (NULL);
}

// Host code
int main(int argc, char** argv)
{
    printf("Salsa20 Vector Encryption using CUDA streams and multi-threaded XOR on CPU\n");
    unsigned int NBLKS = 4000000, N;
    int rv, blks_per_chunk, threadsPerBlock, blocksPerGrid;
    size_t size, i, sz1, sz1_bytes, blk_off;
    unsigned char k[32], *h_A1, *d_A1, *h_B1;
    double gpuTime1, cpuTime1, cpuTime2, strt, en;
    uint64_t v_nonce;
    cudaDeviceProp deviceProp;
    cudaStream_t strm[NUM_STREAMS];
    struct tdat t_data[NUM_STREAMS];

    ParseArguments(argc, argv);
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.major >= 2)
        blks_per_chunk = BLOCKS_PER_CHUNK_2X;
    else
        blks_per_chunk = BLOCKS_PER_CHUNK_1X;

    N = NBLKS / blks_per_chunk;
    if (NBLKS % blks_per_chunk) N++;
    size = NBLKS * XSALSA20_BLOCKSZ;

    /*
     * Create CUDA streams, CPU side threads and initialize thread data.
     */
    for (i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors( cudaStreamCreate(&strm[i]) );
        t_data[i].quit = 0;
        sem_init(&(t_data[i].start), 0, 0);
        sem_init(&(t_data[i].done), 0, 0);
        pthread_create(&(t_data[i].tid), NULL, proc_buf, (void *)&(t_data[i]));
    }

    // Allocate input vectors h_A and h_B in host memory
    pinned = 1;
    if (cudaMallocHost(&h_A, size) != cudaSuccess) {
        pinned = 0;
        if (posix_memalign((void **)&h_A, 16, size) != 0)
            h_A = NULL;
    }
    if (h_A == 0) CleanupResources();
    if ((rv = posix_memalign((void **)&h_B, 16, size)) != 0) {
        if (rv == EINVAL)
            printf("Invalid argument to posix_memalign()\n");
        if (rv == ENOMEM)
            printf("Out of memory\n");
        CleanupResources();
    }

    memset(k, 1, XSALSA20_CRYPTO_KEYBYTES);
    memset(h_nonce, 0, XSALSA20_CRYPTO_NONCEBYTES);

    // Initialize input vectors
    printf("Initializing input data\n");
    Init(h_B, size);

    // Allocate vectors in device memory
    printf("Allocating device buffer\n");
    checkCudaErrors( cudaMalloc((void**)&d_A, size) );

    printf("Starting GPU Calls\n");
    // Copy algorithm parameters to GPU.
    strt = get_wtime_millis();
    checkCudaErrors( cudaMemcpyToSymbol(key, k, XSALSA20_CRYPTO_KEYBYTES, 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(sigma, hsigma, 16, 0, cudaMemcpyHostToDevice) );
    v_nonce = load_littleendian64(h_nonce);
    threadsPerBlock = THREADS_PER_BLOCK;

    /*
     * Launch kernels on the device per stream with overlapped transfer and CPU side processing.
     */
    h_A1 = h_A;
    d_A1 = d_A;
    h_B1 = h_B;
    blk_off = 0;
    for (i = 0; i < NUM_STREAMS; i++) {
        if (i == NUM_STREAMS - 1) {
            sz1 = NBLKS/NUM_STREAMS * NUM_STREAMS;
            sz1 = NBLKS/NUM_STREAMS + (NBLKS - sz1);
        } else {
            sz1 = NBLKS/NUM_STREAMS;
        }
        sz1_bytes = sz1 * XSALSA20_BLOCKSZ;
        N = sz1 / blks_per_chunk;
        if (sz1 % blks_per_chunk) N++;

        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        VecCrypt<<<blocksPerGrid, threadsPerBlock, 0, strm[i%NUM_STREAMS]>>>(d_A1, N, sz1, v_nonce, blks_per_chunk, blk_off);
        checkCudaErrors( cudaMemcpyAsync(h_A1, d_A1, sz1_bytes, cudaMemcpyDeviceToHost, strm[i%NUM_STREAMS]) );
        t_data[i].h_A1 = h_A1;
        t_data[i].h_B1 = h_B1;
        t_data[i].nblks = sz1;
        t_data[i].strm = strm[i%NUM_STREAMS];
        sem_post(&(t_data[i].start));

        h_A1 += sz1_bytes;
        d_A1 += sz1_bytes;
        h_B1 += sz1_bytes;
        blk_off += sz1;
    }

    /*
     * Each thread does a cudaStreamSynchronize() so we do not need a device synchronize here.
     */
    for (i = 0; i < NUM_STREAMS; i++)
        sem_wait(&(t_data[i].done));

    en = get_wtime_millis();
    gpuTime1 = (en - strt);

    // Verify result
    printf("Computing reference code on CPU\n");
    strt = get_wtime_millis();
    crypto_stream_salsa20_ref_xor(h_B, size, h_nonce + 16, k);
    en = get_wtime_millis();
    cpuTime1 = en - strt;
    rv = 0;

    printf("Verifying result\n");
    for (i = 0; i < size; i++) {
	    if (h_B[i] != h_A[i]) {
		    printf("Byte #%llu differ, %d, %d\n", i, h_B[i], h_A[i]);
		    rv = 1;
		    break;
	    }
    }

    printf("Computing optimized code on CPU\n");
    strt = get_wtime_millis();
    crypto_stream_salsa20_amd64_xmm6_xor(h_B, h_B, size, h_nonce + 16, k);
    en = get_wtime_millis();
    cpuTime2 = en - strt;

    /*
     * Clean out keying material on the GPU.
     */
    memset(k, 0, XSALSA20_CRYPTO_KEYBYTES);
    checkCudaErrors( cudaMemcpyToSymbol(key, k, XSALSA20_CRYPTO_KEYBYTES, 0, cudaMemcpyHostToDevice) );
    CleanupResources();

    /*
     * Signal all the threads to quit.
     */
    for (i = 0; i < NUM_STREAMS; i++) {
        t_data[i].quit = 1;
        sem_post(&(t_data[i].start));
        pthread_join(t_data[i].tid, NULL);
    }

    if (pinned)
        printf("Data transfer was pinned\n");
    else
        printf("Data transfer was not pinned\n");
    printf("GPU+CPU computation time (with transfer): %f msec\n", gpuTime1);
    printf("GPU+CPU throughput (with transfer)      : %f MB/s\n", get_mb_s(size, gpuTime1));
    printf("CPU computation time (reference code)   : %f msec\n", cpuTime1);
    printf("CPU throughput (reference code)         : %f MB/s\n", get_mb_s(size, cpuTime1));
    printf("CPU computation time (optimized code)   : %f msec\n", cpuTime2);
    printf("CPU throughput (optimized code)         : %f MB/s\n", get_mb_s(size, cpuTime2));
    if (rv == 0)
        printf("PASSED\n");
    else
        printf("FAILED\n");
}

void CleanupResources(void)
{
    // Free device memory
    if (d_A)
        cudaFree(d_A);

    // Free host memory
    if (h_A) {
        if (pinned)
            cudaFreeHost(h_A);
        else
            free(h_A);
    }
    if (h_B)
        free(h_B);

    cudaDeviceReset();
}

// Allocates an array with random float entries.
void Init(unsigned char *data, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        data[i] = i;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--noprompt") == 0 ||
            strcmp(argv[i], "-noprompt") == 0) 
        {
            noprompt = true;
            break;
        }
    }
}
