#include "Layer.h"
#include "Neuron.h"


void input_layer::add_neurons()
{
	for (int i = 0; i <len; ++i)
	{
		input_neuron *neu = new input_neuron;
		neurons.push_back(neu);
	}
};


void hide_layer::add_neurons()
{
	for (int i = 0; i < len; ++i)
	{
		hide_neuron *neu = new hide_neuron(func);
		neurons.push_back(neu);
	}
};

void hide_layer::activate()
{
	for (int i = 0; i < len; ++i)
	{
		neurons[i]->activate();
	}
};

vector<double> hide_layer::derivative()
{
	vector<double> der;
	for (int i = 0; i < len; ++i)
	{
		der.push_back(neurons[i]->derivative());
	}
	return der;
};




void output_layer::add_neurons()
{
	for (int i = 0; i < len; ++i)
	{
		output_neuron* neu = new output_neuron(func);
		neurons.push_back(neu);
	}
};

void output_layer::activate()
{
	for (int i = 0; i < len; ++i)
	{
		neurons[i]->activate();
	}
};

vector<double> output_layer::derivative()
{
	vector<double> der;
	for (int i = 0; i < len; ++i)
	{
		der.push_back(neurons[i]->derivative());
	}
	return der;
};