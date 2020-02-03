#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "data.h"
#include "cuda_util.h"
#include "kde.h"
#include "sequential_kde.h"
#include "kde_util.h"
#include <chrono>

// Define for profiling
//#define KKI_PROFILE
#include "timer.h"

class cuda_kde : public kde
{
public:
	//Can use onlu one block because of shared memory
	cuda_kde(size_t slices = 4, unsigned threads_per_block = 256, unsigned blocks_on_grid = 8, unsigned shared_memory_threads = 1024)
	:	kde(slices), 
		threads_per_block_(threads_per_block), 
		blocks_on_grid_(blocks_on_grid), 
		shared_memory_threads_(shared_memory_threads)
	{}

	std::vector<std::vector<d_type>> calculate(std::vector<std::vector<d_type>>& input_list) override;

private:
	unsigned threads_per_block_;
	unsigned blocks_on_grid_;
	unsigned shared_memory_threads_;
};

__global__ void shared_triangle_kernel(const d_type* input_data, d_type* output_data, d_type h, size_t size)
{
	//const unsigned stride = blockDim.x * gridDim.x;
	const unsigned stride = blockDim.x;
	//unsigned position = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ d_type shared_memory[];

	// Copy the memory
	for (unsigned position = threadIdx.x; position < size; position += stride)
	{
		shared_memory[position] = input_data[position];
	}

	// Calculate
	for (unsigned position = threadIdx.x; position < size; position += stride)
	{
		d_type weight = 0.;

		for (unsigned i = 0; i < size; ++i)
		{
			// As the memory is sorted, when the index is on an element out of the reach, the loop breaks
			if (shared_memory[position] + h < shared_memory[i]) break;
			const d_type diff = abs(shared_memory[position] - shared_memory[i]);

			if (diff > h) continue;
			weight += (1 - diff / h) / h;
		}

		weight /= size;

		float_t h_opt = h / sqrt(weight);
		weight = 0.;

		for (unsigned i = 0; i < size; ++i)
		{
			if (shared_memory[position] + h_opt < shared_memory[i]) break;
			const d_type diff = abs(shared_memory[position] - shared_memory[i]);

			if (diff > h_opt) continue;
			weight += (1 - diff / h_opt) / h_opt;
		}
		weight /= size;
		output_data[position] = weight;
	}
}

__global__ void triangle_kernel(const d_type* input_data, d_type* output_data, d_type h, size_t size)
{
	//const unsigned stride = blockDim.x * gridDim.x;
	const unsigned stride = blockDim.x;
	//unsigned position = threadIdx.x + blockIdx.x * blockDim.x;
	//extern __shared__ d_type shared_memory[];

	// Copy the memory
	//for (unsigned position = threadIdx.x; position < size; position += stride)
	//{
	//	shared_memory[position] = input_data[position];
	//}

	// Calculate
	for (unsigned position = threadIdx.x; position < size; position += stride)
	{
		d_type weight = 0.;

		for (unsigned i = 0; i < size; ++i)
		{
			// As the memory is sorted, when the index is on an element out of the reach, the loop breaks
			if (input_data[position] + h < input_data[i]) break;
			const d_type diff = abs(input_data[position] - input_data[i]);

			if (diff > h) continue;
			weight += (1 - diff / h) / h;
		}

		weight /= size;

		float_t h_opt = h / sqrt(weight);
		weight = 0.;

		for (unsigned i = 0; i < size; ++i)
		{
			if (input_data[position] + h_opt < input_data[i]) break;
			const d_type diff = abs(input_data[position] - input_data[i]);

			if (diff > h_opt) continue;
			weight += (1 - diff / h_opt) / h_opt;
		}
		weight /= size;
		output_data[position] = weight;
	}
}

std::vector<std::vector<d_type>> cuda_kde::calculate(std::vector<std::vector<d_type>>& input_list)
{
	KKI_PROFILE_FUNCTION();

	using namespace kki;

	KKI_CUDA_ERROR_CHECK("cudaSetDevice", cudaSetDevice(0));
	const d_type k = 1.25;

	/// Create data containers
	std::vector< kki::cuda::cuda_data<d_type>> device_data{ get_number_of_slices() };
	std::vector< kki::cuda::cuda_data<d_type>> device_output{ get_number_of_slices() };
	std::vector<std::vector<d_type>> output_weights{ get_number_of_slices() };

	// Fetch size of shared memory for shared memory heuristic
	int shared_memory_size_;
	cudaDeviceGetAttribute(&shared_memory_size_, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	const unsigned shared_memory_size = static_cast<size_t>(shared_memory_size_);

	// Reserve memory (for one or all slices)
	// Start async data transfer
	for (size_t i = 0; i < input_list.size(); ++i)
	{
		auto& x_slice = input_list[i];
		if (get_region() == get_number_of_slices() || get_region() == i)
		{
			KKI_PROFILE_SCOPE("ALLOCATION");

			const size_t size = x_slice.size() * sizeof(d_type);

			device_data[i].allocate(size);
			KKI_CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(device_data[i], x_slice.data(), size, cudaMemcpyHostToDevice));

			device_output[i].allocate(size);
			output_weights[i].resize(x_slice.size());
		}
	}

	// Launch kernels
	for (size_t i = 0; i < input_list.size(); ++i)
	{
		auto& x_slice = input_list[i];

		if (get_region() == get_number_of_slices() || get_region() == i)
		{
			KKI_PROFILE_SCOPE("Kernel launch");
			try
			{
				//Calculate standard deviation
				d_type h = static_cast<d_type>(k * std::pow(x_slice.size(), -0.2) * standard_deviation(input_list[i]));
				// Calculating size of the slice
				const unsigned mem_size = static_cast<unsigned>(x_slice.size() * sizeof(d_type));
				
				//Launch the kernels
				//	If there is enough shared memory, launch the shared memory version, otherwise, launch the normal one
				if( mem_size <= shared_memory_size)
				{
					shared_triangle_kernel<<<1, shared_memory_threads_, mem_size>>>(device_data[i], device_output[i], h, x_slice.size());
				}
				else
				{
					triangle_kernel<<<blocks_on_grid_, threads_per_block_ >>>(device_data[i], device_output[i], h, x_slice.size());
				}

				// Check for any errors launching the kernel
				cuda::last_error();
			}
			catch (kki::cuda::cuda_error & ex)
			{
				std::cerr << ex.what() << std::endl;
			}
		}
	}

	kki::cuda::device_sync();

	// Transfer data from GPU to CPU
	for (size_t i = 0; i < input_list.size(); ++i)
	{
		auto& x_slice = input_list[i];
		if (get_region() == get_number_of_slices() || get_region() == i)
		{
			KKI_PROFILE_SCOPE("Memcopy");
			try
			{
				cuda::memory_copy(output_weights[i].data(), device_output[i], x_slice.size() * sizeof(d_type), cudaMemcpyDeviceToHost);
			}
			catch (kki::cuda::cuda_error & ex)
			{
				std::cerr << ex.what() << std::endl;
			}
		}
	}

	return output_weights;
}

// Checking function
void test_function_1()
{
	KKI_PROFILE_FUNCTION();

	size_t regions = 4;
	cuda_kde kde_cuda(regions);
	kde_cuda.read_data();
	kde_cuda.load_slices();

	auto cuda = kde_cuda.calculate(kde_cuda.x_par);

	sequential_kde kde_seq(regions);
	kde_seq.read_data();
	kde_seq.load_slices();

	auto seq = kde_seq.calculate(kde_cuda.x_par);

	//Testing
	const d_type epsilon = static_cast<d_type>(0.01);
	for (unsigned region = 0; region < regions; ++region)
	{
		std::cout << "Testing region " << region << ":" << std::endl;

		bool same = true;

		for (size_t i = 0; i < kde_cuda.x_par[region].size(); ++i)
		{
			if (abs(cuda[region][i] - seq[region][i]) > epsilon)
			{
				same = false;
				break;
			}
		}
		std::cout << "\t Test " << (same ? "successful" : "failed") << "!" << std::endl;
	}
}

// Create output files
void test_function_2(std::string file_name)
{
	const size_t regions = 4;

	cuda_kde kde_cuda(regions, 1024, 1);
	kde_cuda.read_data("./data/in/" + file_name + ".txt");
	kde_cuda.load_slices();

	{
		auto cuda = kde_cuda.calculate(kde_cuda.x_par);

		for (unsigned region = 0; region < regions; ++region)
		{
			std::ofstream output_file("./data/out/" + file_name + "/out_par_" + std::to_string(region) + ".txt");

			for (size_t i = 0; i < kde_cuda.x_par[region].size(); ++i)
			{
				output_file << kde_cuda.x_par[region][i] << ' ' << cuda[region][i] << std::endl;
			}

			output_file.close();
		}
	}
	{
		auto cuda = kde_cuda.calculate(kde_cuda.x_apar);

		for (unsigned region = 0; region < regions; ++region)
		{
			std::ofstream output_file("./data/out/" + file_name + "/out_apar_" + std::to_string(region) + ".txt");

			for (size_t i = 0; i < kde_cuda.x_par[region].size(); ++i)
			{
				output_file << kde_cuda.x_par[region][i] << ' ' << cuda[region][i] << std::endl;
			}

			output_file.close();
		}
	}
}

// Thread and block size time analysis 
void test_function_3()
{
	KKI_PROFILE_FUNCTION();

	size_t regions = 4;
	{
		KKI_PROFILE_SCOPE("32_16");
		std::cout << "32_16" << std::endl;


		cuda_kde kde_cuda(regions, 32, 16);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("64_16");
		std::cout << "64_16" << std::endl;


		cuda_kde kde_cuda(regions, 64, 16);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("128_8");
		std::cout << "128_8" << std::endl;

		cuda_kde kde_cuda(regions, 128, 8);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("256_4");
		std::cout << "256_4" << std::endl;

		cuda_kde kde_cuda(regions, 256, 4);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("256_8");
		std::cout << "256_8" << std::endl;

		cuda_kde kde_cuda(regions, 256, 8);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("512_4");
		std::cout << "512_4" << std::endl;


		cuda_kde kde_cuda(regions, 512, 4);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("512_8");
		std::cout << "512_8" << std::endl;


		cuda_kde kde_cuda(regions, 512, 8);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("512_2");
		std::cout << "512_2" << std::endl;


		cuda_kde kde_cuda(regions, 512, 2);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("1024_4");
		std::cout << "1024_4" << std::endl;

		cuda_kde kde_cuda(regions, 1024, 4);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("1024_2");
		std::cout << "1024_2" << std::endl;

		cuda_kde kde_cuda(regions, 1024, 2);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("1024_1");
		std::cout << "1024_1" << std::endl;

		cuda_kde kde_cuda(regions, 1024, 1);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
	{
		KKI_PROFILE_SCOPE("1024_8");
		std::cout << "1024_8" << std::endl;

		cuda_kde kde_cuda(regions, 1024, 8);
		kde_cuda.read_data();
		kde_cuda.load_slices();

		auto cuda = kde_cuda.calculate(kde_cuda.x_par);
	}
}

int main()
{
	KKI_PROFILE_BEGIN("Test");

	//Change to 0 to create output files for all three input files
#if 1

	test_function_1();
	test_function_1();
	test_function_3();
#else
	test_function_2("toyXic2pKpi_200k");
	test_function_2("toyXic2pKpi_200k_KstarAmp5percentage");
	test_function_2("toyXic2pKpi_200k_KstarAmp20percentage");
#endif

	KKI_PROFILE_END();
}
