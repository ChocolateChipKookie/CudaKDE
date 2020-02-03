#pragma once
#include <array>
#include <vector>
#include <random>

//////////////////
///	DATA POINT ///
//////////////////

using d_type = float;

template <size_t Dims, typename  T = d_type>
using data_point = std::array<T, Dims>;


template <size_t Dims>
std::ostream& operator << (std::ostream& stream, const data_point<Dims>& data)
{
	stream << '(';
	for (unsigned i = 0; i < Dims - 1; ++i)
	{
		stream << data[0] << ", ";
	}
	return  stream << data[Dims - 1] << ')';
}
