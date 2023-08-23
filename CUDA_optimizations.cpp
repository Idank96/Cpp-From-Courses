#include <stdio.h>
#include <assert.h>
// ------------------------------------------------ //

// From Loop:

void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  int N = 10;
  loop(N);
}

// To Kernel:

__global__ void loop()
{
  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}


// ------------------------------------------------ //

// From Loop:

void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  int N = 10;
  loop(N);
}

// To Kernel:

__global__ void loop()
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main()
{
  loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}

// ------------------------------------------------ //

// From Loop:

void loop_init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}
// To  Kernel:

__global__ 
void initializeElements(int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    a[i] = i;
  }
}

//From Loop:

void loop_doubleElements(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] *= 2;
  }
}

// To Kernel:

__global__
void doubleElements(int *a, int N)
// Each element in the array is doubled in other thread.
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    a[i] *= 2;
  }
}

// Kernel with stride
__global__
void doubleElementsStride(int *a, int N)
{

  /*
   * Using stride when the number of threads in a grid may
   * be smaller than the size of the data set.
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

inline cudaError_t checkCuda(cudaError_t result)
/*
Error handling in accelerated CUDA code is essential
Memory Managment Functions: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
*/
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{
  int N = 1000;
  int *a;
  size_t size = N * sizeof(int);

  checkCuda( cudaMallocManaged(&a, size) ); 

  /*
  * The following is idiomatic CUDA to make sure there are at
  * least as many threads in the grid as there are `N` elements.
  */
  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  cudaError_t syncErr, asyncErr;

  initializeElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );
  
  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  
  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );
  
  checkCuda( cudaFree(a) ); //free memory that allocated with `cudaMallocManaged`.
}

// ------------------------------------------------ //

// Matrix multiplication in cuda

#define N  64

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N)
  {
    for ( int k = 0; k < N; ++k )
      val += a[row * N + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu;

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}

// ------------------------------------------------ //
// SM - Streaming Multiprocessors 
// properties about the currently active GPU device, including its number of SMs.
#include <stdio.h>

int main()
{
  //Get device ID to query the device.
  int deviceId;
  cudaGetDevice(&deviceId);

  // Get properties about the current device.
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}

// ------------------------------------------------ //
// Adding SM and prefetching
/*
Streaming Multiprocessors
The GPUs that CUDA applications run on have processing units called streaming multiprocessors, or SMs.
During kernel execution, blocks of threads are given to SMs to execute.
In order to support the GPU's ability to perform as many parallel operations as possible, 
performance gains can often be had by choosing a grid size that has a number of blocks that
is a multiple of the number of SMs on a given GPU.
Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called warps.
It is important to know that performance gains can also be had by choosing a block size that
has a number of threads that is a multiple of 32.

Unified Memory Migration
When UM is allocated, the memory is not resident yet on either the host or the device.
When either the host or device attempts to access the memory, a page fault will occur,
at which point the host or device will migrate the needed data in batches. 

Asynchronous Memory Prefetching
A powerful technique to reduce the overhead of page faulting and on-demand memory migrations,
both in host-to-device and device-to-host memory transfers, is called asynchronous memory prefetching.
Using this technique allows programmers to asynchronously migrate unified memory (UM) to any CPU or GPU device in
the system, in the background, prior to its use by application code. 
By doing this, GPU kernels and CPU function performance can be increased on account of reduced page fault and
on-demand data migration overhead.
Prefetching also tends to migrate data in larger chunks, and therefore fewer trips, than on-demand migration.
This makes it an excellent fit when data access needs are known before runtime, 
and when data access patterns are not sparse.
*/
#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

// ------------------------------------------------ //
// SAXPY Exercise 
#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

__global__ void saxpy(int * a, int * b, int * c)
{
  // Determine our unique global thread ID, so we know which element to process
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = tid; i < N; i += stride)
    c[i] = 2 * a[i] + b[i];
}

int main()
{
  int *a, *b, *c;

  int size = N * sizeof (int); // The total number of bytes per vector

  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  // Allocate memory
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // Initialize memory
  for( int i = 0; i < N; ++i )
  {
    a[i] = 2;
    b[i] = 1;
    c[i] = 0;
  }

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  int threads_per_block = 256;
  int number_of_blocks = numberOfSMs * 32;

  saxpy <<<number_of_blocks, threads_per_block>>>( a, b, c );

  cudaDeviceSynchronize(); // Wait for the GPU to finish

  // Print out the first and last 5 values of c for a quality check
  for( int i = 0; i < 5; ++i )
    printf("c[%d] = %d, ", i, c[i]);
  printf ("\n");
  for( int i = N-5; i < N; ++i )
    printf("c[%d] = %d, ", i, c[i]);
  printf ("\n");

  // Free all our allocated memory
  cudaFree( a ); cudaFree( b ); cudaFree( c );
}

// ------------------------------------------------ //
