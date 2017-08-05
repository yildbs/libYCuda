#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <assert.h>

#include <memory>

#include <ycuda/error_util.h>
#include <ycuda/CudaFunctionWrapper.h>

namespace ycuda{

template <typename T>
struct CudaManagedMemory{
private:
	T* buffer;
public:
	CudaManagedMemory(int length) 
		: buffer(nullptr)
	{ 
		checkCudaErrors(CallCudaMallocManaged(&this->buffer, length*sizeof(T)));
		checkCudaErrors(CallCudaDeviceSYnchronize());
	}
	~CudaManagedMemory() 
	{ 
		if (this->buffer != nullptr){
			checkCudaErrors(CallCudaFree(this->buffer));
			this->buffer = nullptr;
		}
	}
	T* const Data() const{
		return this->buffer;
	}
};

template<typename T>
class YUnifiedMemory{
private:
	std::shared_ptr<CudaManagedMemory<T>> data;
	size_t length;
	size_t capacity;

public:
	YUnifiedMemory()
	: data(nullptr)
	, length(0)
	, capacity(0)
	{
	}
	YUnifiedMemory(size_t length)
	: YUnifiedMemory()
	{
		this->Resize(length);
	}
	virtual ~YUnifiedMemory()
	{
	}
	inline T* const Bits(int start_index=0) const
	{
		assert(start_index<this->length && "YUnifiedMemory:: start_index is same or over than length of array");
		assert(start_index>=0 && "YUnifiedMemory:: start_index is less than 0");
		return &this->data.get()->Data()[start_index];
	}
    virtual size_t GetLength() const
    {
        return this->length;
    }
    virtual size_t GetCapacity() const
    {
        return this->capacity;
    }
    virtual YUnifiedMemory<T>& Resize(size_t length){
    	if(this->GetCapacity()>=length){
    		this->length = length;
    	}
    	else{
    		this->length = length;
    		this->capacity = length;
			this->data = std::shared_ptr<CudaManagedMemory<T>>(new CudaManagedMemory<T>(length));
    	}
    	return *this;
    }
    YUnifiedMemory<T>& operator=(YUnifiedMemory<T>&& mem)
	{
		return this->operator =(mem);
	}
    YUnifiedMemory<T>& operator=(YUnifiedMemory<T>& mem)
	{
		//assert(mem.data != nullptr && "YUnifiedMemory:: mem.data must not be nullptr");
		this->data 	= mem.data;
		this->length = mem.length;
		this->capacity = mem.capacity;
		return *this;
	}
    inline YUnifiedMemory<T>& CopyFrom(int start_index, size_t length, T* src)
	{
    	assert(this->length-start_index >= length);
    	memcpy(this->Bits(start_index), src, sizeof(T)*length);
		return *this;
	}
};

}
