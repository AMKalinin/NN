#pragma once

#include <math.h>
#include "Neuron.h"

class neuron;

class function_activation
{
public:
	virtual void activate(neuron &neu) = 0;

	virtual double	 derivative(neuron &neu) = 0;

};

class sigmoida : public function_activation
{
public:
	sigmoida() {};

	void activate(neuron& neu);

	double	 derivative(neuron& neu);
};


class Loss
{
public:
	virtual double loss(std::vector<double> predict, std::vector<double> target) = 0;
	virtual std::vector<double> gradient(std::vector<double> predict, std::vector<double> target) = 0;
};


class SSE : public Loss
{
public:
	SSE() {};
	double loss(std::vector<double> predict, std::vector<double> target);
	std::vector<double> gradient(std::vector<double> predict, std::vector<double> target);
};