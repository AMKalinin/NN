#include <vector>
#include <cstdlib>
#include "Model.h"
#include <iostream>

using std::vector;

void nn::add_layers(layer* lay)
{
	layers.push_back(lay);
};

vector<double> nn::init_w(int size)
{
	vector<double> out;
	for (int i = 0; i < size; ++i)
	{
		out.push_back( ((double)rand()/(double)RAND_MAX)-0.5 );
	};
	return out;
};

void nn::compile()
{
	int sz = layers.size();
	for (int i = 1; i < sz; ++i)
	{
		int ln = layers[i]->len;
		for (int j = 0; j < ln; ++j)
		{	
			for (int k = 0; k < layers[i - 1]->len; ++k)
			{
				layers[i]->neurons[j]->w.push_back(((double)rand() / (double)RAND_MAX) - 0.5);
			};
		};
	};
};

vector<double> nn::forward(vector<double> X)
{
	vector<double> out;
	for (int i = 0; i < (int)X.size(); ++i)
	{
		layers[0]->neurons[i]->out = X[i];
	};
	for (int i = 1; i < (int)layers.size(); ++i)
	{
		for (int j = 0; j < layers[i]->len; ++j)
		{
			layers[i]->neurons[j]->sum = 0;
			for (int k = 0; k < layers[i-1]->len; ++k)
			{
				layers[i]->neurons[j]->sum += layers[i-1]->neurons[k]->out * layers[i]->neurons[j]->w[k];
			};
			layers[i]->neurons[j]->sum += layers[i]->neurons[j]->b;
		};
		layers[i]->activate();
	};
	for (int i = 0; i < layers[layers.size()-1]->len; ++i)
	{
		out.push_back(layers[layers.size() - 1]->neurons[i]->out);
	};
	return out;
};

void nn::backward(vector<double> gradient)
{
	for (int k = layers.size() - 2; k >= 0; --k)
	{
		vector<double> gradientNext;
		vector<double> der = layers[k + 1]->derivative();
		for (int i = 0; i < layers[k + 1]->len; ++i)
			gradient[i] *= der[i] * 0.01; //lr;
		
		for (int i = 0; i < layers[k]->len; ++i)
		{
			gradientNext.push_back(0);
			for (int j = 0; j < layers[k+1]->len; ++j)
				gradientNext[i] += layers[k + 1]->neurons[j]->w[i] * gradient[j];
		};
		
		for (int i = 0; i < layers[k+1]->len; ++i)
		{
			for (int j = 0; j < layers[k]->len; ++j)
				layers[k + 1]->neurons[i]->w[j] -= gradient[i] * layers[k]->neurons[j]->out;
			layers[k + 1]->neurons[i]->b -= gradient[i];
		};
		gradient = gradientNext;
	};
};
