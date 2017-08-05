#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include "CudaFunctionWrapper.h"

namespace ycuda{

cudaError_t CallCudaMemcpy(float* src, float* dst, size_t count, cudaMemcpyKind kind)
{
	return cudaMemcpy(src, dst, count, kind);
}
cudaError_t CallCudaFree(float* ptr)
{
	return cudaFree((void*)ptr);
}
cudaError_t CallCudaMalloc(float** ptr, size_t size)
{
	return cudaMalloc(ptr, size);
}
cudaError_t CallCudaMallocManaged(float** ptr, size_t size)
{
	return cudaMallocManaged((void**)ptr, size);
}
cudaError_t CallCudaMallocManaged(unsigned char** ptr, size_t size)
{
	return cudaMallocManaged((void**)ptr, size);
}
cudaError_t CallCudaDeviceSYnchronize()
{
	return cudaDeviceSynchronize();
}

}
