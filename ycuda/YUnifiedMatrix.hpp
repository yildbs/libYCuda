#include "YUnifiedMemory.hpp"

namespace ycuda{

class YUnifiedMatrix : public YUnifiedMemory<unsigned char>{
	typedef YUnifiedMemory<unsigned char> super;
private:
	size_t width, height, channels;

public:
	YUnifiedMatrix()
	: width(0), height(0), channels(0)
	{
	}
	YUnifiedMatrix(int width, int height=1, int channels=1)
	: width(width), height(height), channels(channels)
	, super::YUnifiedMemory(width*height*channels)
	{
		assert(width>0 && "YUnifiedMatrix:: width is same or less than 0");
		assert(height>0 && "YUnifiedMatrix:: height is same or less than 0");
		assert(channels>0 && "YUnifiedMatrix:: channels is same or less than 0");
	}
	virtual ~YUnifiedMatrix()
	{
	}
	YUnifiedMatrix& operator=(YUnifiedMatrix&& mat)
	{
		YUnifiedMemory::operator =(mat);
		this->width = width;
		this->height = height;
		this->channels = channels;
		return *this;
	}
	inline size_t GetWidth() const{
		return this->width;
	}
	inline size_t GetHeight() const{
		return this->height;
	}
	inline size_t GetChannels() const{
		return this->channels;
	}

};

}
