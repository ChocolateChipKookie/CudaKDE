#pragma once
#include "data.h"


inline d_type triangle_kernel(d_type pos, d_type height)
{
	return height * std::max(1 - abs(pos) * height, static_cast<d_type>(0));
}

inline d_type gaussian_kernel(d_type pos, d_type sigma)
{
	const d_type s2pi = static_cast<const d_type>(1 / std::sqrt(6.28318530718));
	d_type frac = s2pi / sigma;
	d_type p = pos / sigma;
	p *= p;
	d_type ex = static_cast<d_type>(std::exp(-0.5 * p));
	return frac * ex;
}

d_type standard_deviation(const std::vector<float_t>& vec);
