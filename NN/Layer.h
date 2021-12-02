#pragma once
#include <vector>
#include "Neuron.h"
#include "Functions.h"
using std::vector;
class function_activation;
class neuron;

class layer
{
public:
	vector<neuron*> neurons;
	int len = 0;
	virtual void add_neurons() = 0;
	virtual void activate() = 0;
	virtual vector<double> derivative() = 0;
};


class input_layer: public layer
{
public:
	input_layer(int size_layer) 
	{	
		len = size_layer;
		add_neurons();
	}
	void add_neurons();
	void activate() {};
	vector<double> derivative() { vector<double> m = { 0 }; return m; };


};

class hide_layer : public layer
{
public:
	function_activation *func;
	hide_layer(int size_layer, function_activation* funct)
	{
		len = size_layer;
		func = funct;
		add_neurons();
	};
	void add_neurons();
	void activate();
	vector<double> derivative();

};


class output_layer : public layer
{
public:
	function_activation* func;
	output_layer(int size_layer, function_activation* funct)
	{
		len = size_layer;
		func = funct;
		add_neurons();
	};
	void add_neurons();
	void activate();
	vector<double> derivative();

};