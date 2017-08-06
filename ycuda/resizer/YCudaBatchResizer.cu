#include <iostream>
#include <chrono>
#include <time.h>

#include <ycuda/resizer/YCudaBatchResizer.h>

namespace ycuda{
namespace resizer{

__global__ void CudaKernel_BatchResize_GRAY2GRAY(
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
	value += (float)src_image[src_width*(src_y + coor_y_in_rect + 0) + (src_x + coor_x_in_rect + 0)] * fx * fy;
	value += (float)src_image[src_width*(src_y + coor_y_in_rect + 0) + (src_x + coor_x_in_rect + 1)] * (1.0f - fx)*fy;
	value += (float)src_image[src_width*(src_y + coor_y_in_rect + 1) + (src_x + coor_x_in_rect + 0)] * fx*(1.0f - fy);
	value += (float)src_image[src_width*(src_y + coor_y_in_rect + 1) + (src_x + coor_x_in_rect + 1)] * (1.0f - fx)*(1.0f - fy);

	dst_ptr[blockIdx.x * blockDim.x + threadIdx.x] = value / 255.f;
}

/**
 * YCudaBatchMatrix
 */
YCudaBatchMatrix::YCudaBatchMatrix()
: num_matrix(10)
, width(0)
, height(0)
, channels(1)
, initialized(false)
{
}
YCudaBatchMatrix& YCudaBatchMatrix::SetSize(int width, int height, int channels)
{
	assert(width>0 && "YCudaBatchMatrix:: width is same or less than 0");
	assert(height>0 && "YCudaBatchMatrix:: height is same or less than 0");
	assert(channels>0 && "YCudaBatchMatrix:: channels is same or less than 0");
	assert(initialized==false && "YCudaBatchMatrix:: Try to SetSize function after initialized");
	this->initialized = true;
	this->width = width;
	this->height = height;
	this->channels = channels;
	this->SetNumMatrix(this->num_matrix);
	return *this;
}
YCudaBatchMatrix& YCudaBatchMatrix::SetNumMatrix(size_t num_matrix)
{
	assert(this->initialized==true && "YCudaBatchMatrix:: Call SetNumMatrix function before initialized");
	assert(num_matrix>0 && "YCudaBatchMatrix:: num_matrix is same or less than 0");
	int length_elem = this->width * this->height * this->channels;
	this->num_matrix = num_matrix;
	this->data.Resize(num_matrix*length_elem);
	return *this;
}
inline int YCudaBatchMatrix::GetWidth() const
{
	return this->width;
}
inline int YCudaBatchMatrix::GetHeight() const
{
	return this->height;
}
inline size_t YCudaBatchMatrix::GetLength() const
{
	return this->num_matrix*this->width*this->height*this->channels;
}
float* const YCudaBatchMatrix::Bits() const
{
	assert(this->initialized==true && "YCudaBatchMatrix:: Get the pointer before initialized");
	return data.Bits();
}
YUnifiedMemory<float>& YCudaBatchMatrix::GetData()
{
	return this->data;
}
/**
 * YListRect
 */
YListRect::YListRect(int capacity)
:num_rects(capacity), start_index(0)
{
	this->SetNumRects(capacity);
}
YListRect& YListRect::SetNumRects(int capacity)
{
	this->num_rects = capacity;
	this->data.Resize(capacity*4);
	this->Reset();
	return *this;
}
int YListRect::GetNumRects() const
{
	return this->start_index;
}
YListRect& YListRect::Reset()
{
	this->start_index = 0;
	return *this;
}
int YListRect::PushRect(int x, int y, int w, int h)
{
	assert(this->num_rects > this->start_index && "YListRect:: start_index must be lower than capacity");
	data.Bits(start_index*4)[0] = x;
	data.Bits(start_index*4)[1] = y;
	data.Bits(start_index*4)[2] = w;
	data.Bits(start_index*4)[3] = h;
	return this->start_index++;
}
size_t YListRect::GetLength() const
{
	return this->start_index;
}
int* const YListRect::Bits() const
{
	return this->data.Bits();
}


/**
 * YCudaBatchResizer
 */
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
	//assert(type==MatrixType::GRAY && "YCudaBatchResizer:: Only resizing GRAY image is implemented");
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
	//assert(type==MatrixType::GRAY && "YCudaBatchResizer:: Only resizing GRAY image is implemented");
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

#define CALCULATE_CUDA_OCCUPANCY 0
#define MEASURE_TIME 0

size_t YCudaBatchResizer::CudaBatchResize(int size, unsigned char* ptr)
{
	//Copy source image
	src.CopyFrom(0, size, ptr);

	this->dst.SetNumMatrix(this->rects.GetNumRects());

#if CALCULATE_CUDA_OCCUPANCY
	//For estimating cuda occupancy
	int min_grid_size, grid_size;
	int optimal_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size, CudaKernel_BatchResize_GRAY2GRAY, 0, 0);
	printf("min_grid_size : %d\n", min_grid_size);
	printf("optimal_block_size : %d\n", optimal_block_size);
	//-----------------------------
#endif

#if MEASURE_TIME
	auto start_resizer = std::chrono::high_resolution_clock::now();
#endif
	if(src_type==MatrixType::GRAY && dst_type==MatrixType::GRAY){
		int num_total_threads = this->dst.GetLength();
		int block_size = 640;
#if CALCULATE_CUDA_OCCUPANCY
		block_size = optimal_block_size;
#endif
		dim3 dim_block(block_size);
		dim3 dim_grid(num_total_threads%block_size==0? num_total_threads/block_size : static_cast<int>(num_total_threads/block_size)+1);
		CudaKernel_BatchResize_GRAY2GRAY<<<dim_grid, dim_block>>>(
			this->src.GetWidth(),
			this->src.Bits(),
			this->rects.GetNumRects(),
			this->rects.Bits(),
			this->dst.GetWidth(),
			this->dst.GetHeight(),
			this->dst.Bits()
		);
	} 
	else{
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
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, CudaKernel_BatchResize_GRAY2GRAY, optimal_block_size, 0);
	printf("max_active_blocks : %d\n", max_active_blocks);
	int device;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	printf("props.warpSize : %d\n", props.warpSize);
	printf("props.maxThreadsPerMultiProcessor : %d\n", props.maxThreadsPerMultiProcessor);
	printf("optimal_block_size : %d\n", optimal_block_size);
	float occupancy = (float)((float)((float)max_active_blocks*(float)optimal_block_size/(float)props.warpSize) / (float)((float)props.maxThreadsPerMultiProcessor/(float)props.warpSize));
	printf("occupancy : %.2f\n", occupancy);
	//-----------------------------
#endif

	return this->rects.GetNumRects();
}

float* const YCudaBatchResizer::GetDstBits() const
{
	return this->dst.Bits();
}
YUnifiedMemory<float>& YCudaBatchResizer::GetDst()
{
	return this->dst.GetData();
}

}
}
