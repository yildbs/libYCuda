#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

namespace ycuda{

	template<typename T>
	extern cudaError_t CallCudaMemcpy(T* src, T* dst, size_t count, cudaMemcpyKind kind);
	template<typename T>
	extern cudaError_t CallCudaMalloc(T** ptr, size_t size);
	template<typename T>
	extern cudaError_t CallCudaFree(T* ptr);
	template<typename T>
	extern cudaError_t CallCudaMallocManaged(T** ptr, size_t size);
	cudaError_t CallCudaDeviceSYnchronize();

}
