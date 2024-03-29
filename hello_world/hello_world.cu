#include <iostream> 
#include <vector>
#include <cuda.h>

#include <nvToolsExt.h>

struct Tracer 
{
    Tracer(const char* name) {
        nvtxRangePushA(name);
    }
    ~Tracer() {
        nvtxRangePop();
    }
};

#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a ## b)
#define PP_CAT_II(p, res) res
#define UNIQUE_NAME(base) PP_CAT(base, __COUNTER__)
#define TRACER(name) Tracer UNIQUE_NAME(name)(#name)

__global__ void compute_kernel(int* data)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    data[tid] = tid;
    if (tid == 0) printf("hello world from kernel!\n");
}

int main()
{
    size_t size = 4096;
    int* d_data = nullptr;
    {
        TRACER(malloc);
        cudaMalloc(&d_data, sizeof(int)*size);
    }
    {
        TRACER(kernelLaunch);
        size_t blockSize = 32;
        size_t gridSize = 128;
        compute_kernel<<<blockSize, gridSize>>>(d_data);
        cudaStreamSynchronize(0);
    }
    {
        TRACER(free);
        cudaFree(d_data);
    }
    return 0;
} 