#pragma once
#include "kde.h"

class sequential_kde : public kde
{
public:
	sequential_kde(size_t slices) : kde(slices){}

	std::vector<std::vector<d_type>> calculate(std::vector<std::vector<d_type>>& input_list) override;
};

