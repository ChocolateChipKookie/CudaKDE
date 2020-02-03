#include "sequential_kde.h"
#include "kde_util.h"

std::vector<std::vector<d_type>> sequential_kde::calculate(std::vector<std::vector<d_type>>& input_list)
{
	std::vector<d_type> wsk;
	d_type k = 1.25;
	std::vector<std::vector<d_type>> output_weights{get_number_of_slices()};

	for (size_t i = 0; i < input_list.size(); ++i)
	{
		auto& x_slice = input_list[i];
		output_weights[i].reserve(x_slice.size());

		if (get_region() == get_number_of_slices() || get_region() == i)
		{
			d_type h = static_cast<d_type>(k * std::pow(x_slice.size(), -0.2) * standard_deviation(input_list[i]));
			for (auto xi : x_slice)
			{
				d_type weight = 0.;

				switch (get_function())
				{
				case kernel::triangle:
					{
						for (const auto x : x_slice)
						{
							d_type abs_dx = std::abs(x - xi);

							if (abs_dx < h)
							{
								weight += (1 - abs_dx / h) / h;
							}
						}
						weight /= x_slice.size();
						
						float_t h_opt = h / sqrt(weight);
						weight = 0.;

						for (const auto x : x_slice)
						{
							float_t abs_dx = std::abs(x - xi);

							if (abs_dx < h_opt)
							{
								weight += (1 - abs_dx / h_opt) / h_opt;
							}
						}
						weight /= x_slice.size();
						
						break;
					}
				case kernel::gauss:
					{
						for (const auto x : x_slice)
						{
							weight += gaussian_kernel(x - xi, h);
						}
						weight /= x_slice.size();

						d_type h_opt = h / sqrt(weight);
						weight = 0.;
						for (const auto x : x_slice)
						{
							weight += gaussian_kernel(x - xi, h_opt);
						}
						weight /= x_slice.size();
					}
					break;
				default:
					throw std::runtime_error{"Kernel not supported!"};
				}
				output_weights[i].push_back(weight);
			}
		}
	}
	return output_weights;
}
