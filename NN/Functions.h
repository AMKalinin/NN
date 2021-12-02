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