#include "YCudaBatchResizer.h"
#include <iostream>
#include <chrono>
#include <time.h>

namespace ydnn{
namespace cuda{
namespace resizer{


__global__ void CudaKernel_BatchResize_Gray2Gray(
		int src_width,
		unsigned char* src_image,
		int num_rects,
		int* rects,
		int dst_width,
		int dst_height,
		float* dst_ptr
)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_image_size = dst_width * dst_height;
	if( num_rects*dst_image_size <= gid ){
		return;
	}

	const int image_index = (int)(gid / dst_image_size);
	const int pixel_index = gid % dst_image_size;

	float scale_x = (float)(rects[image_index*4 + 2])/dst_width;
	float fx = (float)(((pixel_index % dst_width)+0.5f)*scale_x - 0.5);
	int coor_x_in_rect = floor(fx);
	fx = 1.0f - (fx - (float)coor_x_in_rect);

	float scale_y = (float)(rects[image_index*4 + 3])/dst_height;
	float fy = (float)(((pixel_index / dst_width)+0.5f)*scale_y - 0.5);
	int coor_y_in_rect = floor(fy);
	fy = 1.0f - (fy - (float)coor_y_in_rect);

	int src_x = rects[image_index*4 + 0];
	int src_y = rects[image_index*4 + 1];

	float value = 0.;
	value += (float)src_image[src_width*(src_y+coor_y_in_rect+0) + (src_x+coor_x_in_rect+0)] * fx * fy;
	value += (float)src_image[src_width*(src_y+coor_y_in_rect+0) + (src_x+coor_x_in_rect+1)] * (1.0f-fx)*fy;
	value += (float)src_image[src_width*(src_y+coor_y_in_rect+1) + (src_x+coor_x_in_rect+0)] * fx*(1.0f-fy);
	value += (float)src_image[src_width*(src_y+coor_y_in_rect+1) + (src_x+coor_x_in_rect+1)] * (1.0f-fx)*(1.0f-fy);

	dst_ptr[blockIdx.x * blockDim.x + threadIdx.x] = value / 255.f;
}

YCudaBatchResizer::YCudaBatchResizer()
: src_type(MatrixType::NOT_INITIALIZED)
, dst_type(MatrixType::NOT_INITIALIZED)
, src_width(0)
, src_height(0)
, initialized(false)
{}

YCudaBatchResizer::~YCudaBatchResizer()
{}

YCudaBatchResizer& YCudaBatchResizer::SetSourceSize(size_t width, size_t height, MatrixType type)
{
	//TODO
	assert(type==MatrixType::GRAY && "YCudaBatchResizer:: Only resizing GRAY image is implemented");
	assert(width>0 && "YUnifiedMatrix:: width is same or less than 0");
	assert(height>0 && "YUnifiedMatrix:: height is same or less than 0");
	this->src_type = type;
	this->src = YUnifiedMatrix(width, height, static_cast<int>(type));
	this->src_width = width;
	this->src_height = height;
	return *this;
}
YCudaBatchResizer& YCudaBatchResizer::SetDestinationSize(size_t width, size_t height, MatrixType type)
{
	//TODO
	assert(this->initialized==false && "YCudaBatchResizer:: Reinitialize the destination size is not allowed");
	assert(this->src_type!=MatrixType::NOT_INITIALIZED && "Src type is not initialized");
	assert(type==MatrixType::GRAY && "YCudaBatchResizer:: Only resizing GRAY image is implemented");
	assert(width>0 && "YUnifiedMatrix:: width is same or less than 0");
	assert(height>0 && "YUnifiedMatrix:: height is same or less than 0");
	this->dst_type = type;
	this->dst.SetSize(width, height, static_cast<int>(type));
	this->initialized = true;
	return *this;
}

YCudaBatchResizer& YCudaBatchResizer::SetNumMatrix(size_t num_matrix)
{
	this->rects.SetNumRects(num_matrix);
	this->dst.SetNumMatrix(num_matrix);
	return *this;
}

YCudaBatchResizer& YCudaBatchResizer::ResetRects()
{
	this->rects.Reset();
	return *this;
}

YCudaBatchResizer& YCudaBatchResizer::PushRect(int x, int y, int w, int h)
{
	assert(x+w < this->src_width && "x+w must be lower than dst_width");
	assert(y+h < this->src_height && "y+h must be lower than dst_height");
	int current_index = this->rects.PushRect(x, y, w, h);
	return *this;
}


#define CALCULATE_CUDA_OCCUPANCY 1
#define MEASURE_TIME 1

size_t YCudaBatchResizer::CudaBatchResize(int size, unsigned char* ptr)
{
	//Copy source image
	src.CopyFrom(0, size, ptr);

	int num_total_threads = this->dst.GetLength();
	int block_size = 640;

#if CALCULATE_CUDA_OCCUPANCY
	//For estimating cuda occupancy
	int min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, CudaKernel_BatchResize_Gray2Gray, 0, 0);
	printf("min_grid_size : %d\n", min_grid_size);
	printf("block_size : %d\n", block_size);
	//-----------------------------
#endif

#if MEASURE_TIME
	auto start_resizer = std::chrono::high_resolution_clock::now();
#endif

	dim3 dim_block(block_size);
	dim3 dim_grid(num_total_threads%block_size==0? num_total_threads/block_size : static_cast<int>(num_total_threads/block_size)+1);
	if(src_type==MatrixType::GRAY && dst_type==MatrixType::GRAY){
		CudaKernel_BatchResize_Gray2Gray<<<dim_grid, dim_block>>>(
			this->src.GetWidth(),
			this->src.Bits(),
			this->rects.GetNumRects(),
			this->rects.Bits(),
			this->dst.GetWidth(),
			this->dst.GetHeight(),
			this->dst.Bits()
		);
	}else{
		assert(0 && "YCudaBatchResizer:: Not implemented resizing function");
	}
	cudaDeviceSynchronize();

#if MEASURE_TIME
	std::chrono::duration<double> elapsed_start_resizer = std::chrono::high_resolution_clock::now() - start_resizer;
	std::cout << "Total elapsed time ! : " << elapsed_start_resizer.count() << std::endl;
#endif

#if CALCULATE_CUDA_OCCUPANCY
	//For estimating cuda occupancy
	int max_active_blocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, CudaKernel_BatchResize_Gray2Gray, block_size, 0);
	printf("max_active_blocks : %d\n", max_active_blocks);
	int device;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	printf("props.warpSize : %d\n", props.warpSize);
	printf("props.maxThreadsPerMultiProcessor : %d\n", props.maxThreadsPerMultiProcessor);
	printf("block_size : %d\n", block_size);
	float occupancy = (float)((float)((float)max_active_blocks*(float)block_size/(float)props.warpSize) / (float)((float)props.maxThreadsPerMultiProcessor/(float)props.warpSize));
	printf("occupancy : %.2f\n", occupancy);
	//-----------------------------
#endif

	return this->rects.GetNumRects();
}

float* YCudaBatchResizer::GetDstBits() const
{
	return this->dst.Bits();
}

}
}
}
