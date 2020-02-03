#pragma once
#include "kde_util.h"
#include <numeric>

d_type standard_deviation(const std::vector<d_type>& vec)
{
	const d_type sum = std::accumulate(vec.begin(), vec.end(), static_cast<d_type>(0));
	const d_type mean = sum / static_cast<d_type>(vec.size());
	d_type standard_dev{ 0. };
	for (size_t i = 0; i < vec.size(); ++i)
	{
		const d_type diff = vec[i] - mean;
		standard_dev += diff * diff;
	}

	return sqrt(standard_dev / vec.size());
}
