#pragma once
#include "kde.h"
#include <iostream>
#include <sstream>
#include <fstream>

kde::~kde() = default;
kde::kde(size_t slices) : x_par(slices), x_apar(slices), region_(slices), slices_(slices){}

void kde::read_data(const std::string& input)
{
	std::ifstream input_file(input);
	std::string s;

	std::getline(input_file, s);
	const size_t entries = std::stoi(s);
	vec_id_.reserve(entries);
	vec_m2pk_.reserve(entries);
	vec_m2kpi_.reserve(entries);

	long long id;
	d_type m, mpi;

	for (size_t i = 0; i < entries; ++i)
	{
		if (!std::getline(input_file, s))return;
		std::stringstream stream{s};
		stream >> id >> m >> mpi;
		vec_id_.push_back(id);
		vec_m2pk_.push_back(m);
		vec_m2kpi_.push_back(mpi);
	}
}

void kde::load_slices()
{
	auto y_max = *std::max_element(vec_m2kpi_.begin(), vec_m2kpi_.end()) + 0.01;
	auto y_min = *std::min_element(vec_m2kpi_.begin(), vec_m2kpi_.end()) - 0.01;
	auto y_bin = (y_max - y_min) / slices_;

	std::vector<d_type> y_range;
	y_range.reserve(slices_);

	for (size_t s = 0; s < slices_; ++s)
	{
		y_range.push_back(y_min + y_bin * s);
	}

	for (size_t i = 0; i < vec_m2kpi_.size(); ++i)
	{
		auto& y = vec_m2kpi_[i];

		for (size_t j = 0; j < y_range.size(); ++j)
		{
			auto& dy = y_range[j];
			if (y > dy && y < dy + y_bin)
			{
				if (vec_id_[i] == 1)
				{
					x_par[j].push_back(vec_m2pk_[i]);
					break;
				}
				else
				{
					x_apar[j].push_back(vec_m2pk_[i]);
					break;
				}
			}
		}
	}

	for (size_t i = 0; i < slices_; ++i)
	{
		std::sort(x_par[i].begin(), x_par[i].end());
		std::sort(x_apar[i].begin(), x_apar[i].end());
	}

	size_t total = 0;
	for (size_t i = 0; i < slices_; ++i)
	{
		total += x_par[i].size() + x_apar[i].size();
	}

	if (total != vec_m2kpi_.size())
		throw std::runtime_error{"Sum of particles and antiparticles not equal to total!"};
}

void kde::set_region(size_t region) const
{
	if (region < slices_)
	{
		region = region;
	}
	else
	{
		region = slices_;
	}
}

void kde::set_function(kernel function)
{
	function_ = function;
}
