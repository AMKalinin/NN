#include "Functions.h"

void sigmoida::activate(neuron& neu)
{
	neu.out = 1 / (1 + exp(-(neu.sum)));
};

double	sigmoida::derivative(neuron& neu)
{
	return (neu.out * (1 - neu.out));
};