#pragma once
#include "data.h"
#include "cuda_runtime.h"
#include <string>
#include <cassert>

/// Dont put the function in the place of status, because it will be executed twice
#define KKI_CUDA_ERROR_CHECK(name, status) {				\
	cudaError_t tmp_status = (status);						\
	if (tmp_status != cudaSuccess)							\
		throw kki::cuda::cuda_error{ (name), tmp_status };	\
	}														\

namespace kki
{
	/// Data and function wrappers for the cuda library
	/// 
	namespace cuda
	{
		class cuda_error : public std::runtime_error
		{
		public:
			cuda_error(const std::string& function, cudaError_t error)
				: std::runtime_error(function + "failed with (" + std::to_string(error) + "){" + cudaGetErrorString(error) + "}") {}
			cuda_error(const std::string& what_)
				: std::runtime_error(what_) {}
		};

		template <typename T = d_type>
		class cuda_data
		{
		public:
			explicit cuda_data(){}

			explicit cuda_data(size_t size)
			{
				const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&mem_ptr_), size);
				if (status != cudaSuccess)
				{
					throw cuda_error{ "cudaMalloc", status };
				}
			}
			~cuda_data()
			{
				cudaFree(mem_ptr_);
			}

			void allocate(size_t size)
			{
				assert(mem_ptr_ != nullptr);
				const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&mem_ptr_), size);
				if (status != cudaSuccess)
				{
					throw cuda_error{ "cudaMalloc", status };
				}
			}

			operator T* ()
			{
				assert(mem_ptr_ != nullptr);
				return mem_ptr_;
			}

			T* ptr() const
			{
				assert(mem_ptr_ != nullptr);
				return mem_ptr_;
			}

		private:
			T* mem_ptr_{ nullptr };
		};

		inline void memory_copy(void* dst, const void* src, size_t size, cudaMemcpyKind kind)
		{
			KKI_CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(dst, src, size, kind));
		}



		inline void device_sync()
		{
			const cudaError_t status = cudaDeviceSynchronize();
			KKI_CUDA_ERROR_CHECK("cudaDeviceSynchronize", status);
		}

		inline void last_error()
		{
			const cudaError_t status = cudaGetLastError();
			if (status != cudaSuccess) {
				throw cuda_error{ std::string{"Last error: ("} +std::to_string(status) + "){" + cudaGetErrorString(status) + "}" };
			}
		}

		inline void last_error(const std::string& function)
		{
			const cudaError_t status = cudaGetLastError();
			KKI_CUDA_ERROR_CHECK(function, status);
		}
	}

	template <typename T>
	class uninitialized_vec
	{
	public:

		uninitialized_vec(size_t size) : mem_ptr_(new T[size]), end_(mem_ptr_ + size), size_(size){}
		
		~uninitialized_vec()
		{
			delete[]mem_ptr_;
		}	

		void allocate(size_t size)
		{
			assert(mem_ptr_ != nullptr);
			mem_ptr_ = new T[size];
			end_ = mem_ptr_ + size;
			size_ = size;
		}

		T* begin()
		{
			assert(mem_ptr_ != nullptr);

			return mem_ptr_;
		}
		
		T* end()
		{
			assert(mem_ptr_ != nullptr);
			return end_;
		}
		
		T& operator[](size_t pos)
		{
			assert(mem_ptr_ != nullptr);
			return mem_ptr_[pos];
		}
		
		T* data()
		{
			assert(mem_ptr_ != nullptr);
			return mem_ptr_;
		}
		
		size_t size() const
		{
			return size_;
		}

	private:
		T* mem_ptr_{ nullptr };
		T* end_{ nullptr };
		size_t size_{ 0 };
	};
}


