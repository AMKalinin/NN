#include "Neuron.h"

hide_neuron::hide_neuron(function_activation* funct) { func = funct; };

void hide_neuron::activate()
{
	func->activate(*this);
};

double hide_neuron::derivative()
{
	return func->derivative(*this);
};

output_neuron::output_neuron(function_activation* funct):hide_neuron(funct) { };
