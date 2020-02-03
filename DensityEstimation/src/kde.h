#pragma once
#include "data.h"
#include <string>

class kde
{
public:
	enum class kernel { triangle, gauss };

	kde(size_t slices = 4);
	virtual ~kde();

	void read_data(const std::string& input = "./data/in/toyXic2pKpi_200k.txt");
	void load_slices();

	size_t get_number_of_slices() const { return slices_; }

	void set_region(size_t region) const;
	size_t get_region() const { return region_; }

	void set_function(kernel function);
	kernel get_function() const { return function_; };

	virtual std::vector<std::vector<d_type>> calculate(std::vector<std::vector<d_type>>& input_list) = 0;

	std::vector<std::vector<d_type>> x_par;
	std::vector<std::vector<d_type>> x_apar;
private:
	size_t region_;
	kernel function_{ kernel::triangle };
	size_t slices_;
	std::vector<long long> vec_id_;
	std::vector<d_type> vec_m2pk_;
	std::vector<d_type> vec_m2kpi_;
};
