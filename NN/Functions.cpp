#include "Functions.h"

void sigmoida::activate(neuron& neu)
{
	neu.out = 1 / (1 + exp(-(neu.sum)));
};

double	sigmoida::derivative(neuron& neu)
{
	return (neu.out * (1 - neu.out));
};


double SSE::loss(vector<double> predict, vector<double> target)
{
	double squared_errors = 0;
	for (int i = 0; i < predict.size(); i++)
		squared_errors += pow((predict[i] - target[i]), 2);
	return squared_errors;
};

vector<double> SSE::gradient(vector<double> predict, vector<double> target)
{
	vector<double> grad;
	for (int i = 0; i < predict.size(); i++)
		grad.push_back(2 * (predict[i] - target[i]));
	return grad;
};