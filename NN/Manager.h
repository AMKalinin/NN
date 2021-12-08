#pragma once
#include "Model.h"
#include "Functions.h"

class manager
{
public:
	model* md;
	double lr;
	Loss* loss;

	manager(model* mod, double learning_rate, Loss* loss_function);
	void train(vector<vector<double>> x_train, vector<vector<double>> y_train);
	void test(vector<vector<double>> x_test, vector<vector<double>> y_test);
	void fit(vector<vector<double>> x_train, vector<vector<double>> y_train, vector<vector<double>> x_test, vector<vector<double>> y_test, int epoch);
};