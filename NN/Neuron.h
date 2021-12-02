#pragma once
#include <vector>
#include "Functions.h"
using std::vector;

class function_activation;

class neuron
{
public:
	vector<double> w;
	double out = 0;
	double sum = 0;
	double b = 0;

	neuron()
	{
		out = 0;
	};

	virtual void activate() = 0;
	virtual double derivative() = 0;
};


class input_neuron : public neuron
{
public:
	input_neuron():neuron()
	{
	};
	void activate() {  };
	double derivative() { return 0; };
};

class hide_neuron : public neuron
{	
public:
	function_activation *func;
	
	hide_neuron(function_activation *funct);
	void activate();
	double derivative();
};


class output_neuron : public hide_neuron
{
public:
	output_neuron(function_activation* funct);
};