#include <vector>
#include <cuda.h>

#include <ycuda/YUnifiedMatrix.hpp>

namespace ycuda{
namespace resizer{

class YCudaBatchMatrix{
private:
	bool initialized;
	size_t num_matrix;
	int width, height, channels;
	YUnifiedMemory<float> data;
public:
	YCudaBatchMatrix();
	YCudaBatchMatrix& SetSize(int width, int height, int channels = 1);
	YCudaBatchMatrix& SetNumMatrix(size_t num_matrix);
	inline int GetWidth() const;
	inline int GetHeight() const;
	inline size_t GetLength() const;
	float* const Bits() const;
	YUnifiedMemory<float>& GetData();
};

class YListRect{
private:
	int num_rects;
	int start_index;
	YUnifiedMemory<int> data;
public:
	YListRect(int capacity = 10);
	YListRect& SetNumRects(int capacity);
	int GetNumRects() const;
	YListRect& Reset();
	int PushRect(int x, int y, int w, int h);
	size_t GetLength() const;
	int* const Bits() const;
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

	float* const GetDstBits() const;
	YUnifiedMemory<float>& GetDst();
};


}
}
