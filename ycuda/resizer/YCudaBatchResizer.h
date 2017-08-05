#include <vector>

#include <cuda.h>

//#include "../../../libYCV/YCVCore/core.hpp"
#include "../YUnifiedMatrix.hpp"

namespace ydnn{
namespace cuda{
namespace resizer{

class YCudaBatchMatrix{
private:
	bool initialized;
	size_t num_matrix;
	int width, height, channels;
	YUnifiedMemory<float> dst;
public:
	YCudaBatchMatrix()
	: num_matrix(10)
	, width(0)
	, height(0)
	, channels(1)
	, initialized(false)
	{}
	YCudaBatchMatrix& SetSize(int width, int height, int channels=1)
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
	YCudaBatchMatrix& SetNumMatrix(size_t num_matrix)
	{
		assert(this->initialized==true && "YCudaBatchMatrix:: Call SetNumMatrix function before initialized");
		assert(num_matrix>0 && "YCudaBatchMatrix:: num_matrix is same or less than 0");
		int length_elem = this->width * this->height * this->channels;
		this->num_matrix = num_matrix;
		this->dst.Resize(num_matrix*length_elem);
		return *this;
	}
	inline int GetWidth() const
	{
		return this->width;
	}
	inline int GetHeight() const
	{
		return this->height;
	}
	inline size_t GetLength() const
	{
		return this->num_matrix*this->width*this->height*this->channels;
	}
	float* Bits() const
	{
		assert(this->initialized==true && "YCudaBatchMatrix:: Get the pointer before initialized");
		return dst.Bits();
	}
};

class YListRect{
private:
	int num_rects;
	int start_index;
	YUnifiedMemory<int> data;
public:
	YListRect(int capacity=10)
	:num_rects(capacity), start_index(0)
	{
		this->SetNumRects(capacity);
	}
	YListRect& SetNumRects(int capacity)
	{
		this->num_rects = capacity;
		this->data.Resize(capacity*4);
		this->Reset();
		return *this;
	}
	int GetNumRects() const
	{
		return this->start_index;
	}
	YListRect& Reset()
	{
		this->start_index = 0;
		return *this;
	}
	int PushRect(int x, int y, int w, int h)
	{
		assert(this->num_rects > this->start_index && "YListRect:: start_index must be lower than capacity");
		data.Bits(start_index*4)[0] = x;
		data.Bits(start_index*4)[1] = y;
		data.Bits(start_index*4)[2] = w;
		data.Bits(start_index*4)[3] = h;
		return this->start_index++;
	}
	size_t GetLength() const
	{
		return this->start_index;
	}
	int* Bits() const
	{
		return this->data.Bits();
	}
};



class YCudaBatchResizer{
public:
	enum MatrixType{
		NOT_INITIALIZED=0,
		GRAY=1,
		RGB=3,
		RGBA=4
	};
private:
	bool initialized;

	//SOURCE
	MatrixType src_type;
	int src_width, src_height;
	YUnifiedMatrix src;
	YListRect rects;

	//DESTINATION
	MatrixType dst_type;
	YCudaBatchMatrix dst;


public:
	YCudaBatchResizer();
	virtual ~YCudaBatchResizer();
	YCudaBatchResizer& SetSourceSize(size_t width, size_t height, MatrixType type=MatrixType::GRAY);
	YCudaBatchResizer& SetDestinationSize(size_t width, size_t height, MatrixType type=MatrixType::GRAY);

	YCudaBatchResizer& SetNumMatrix(size_t num_matrix);
	YCudaBatchResizer& ResetRects();
	YCudaBatchResizer& PushRect(int x, int y, int w, int h);

	size_t CudaBatchResize(int size, unsigned char* ptr);
	float* GetDstBits() const;
};


}
}
}
