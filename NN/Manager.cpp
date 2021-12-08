#include "Manager.h"
#include <iostream>


manager::manager(model* mod, double learning_rate, Loss* loss_function)
{
	md = mod;
	lr = learning_rate;
	md->lr = lr;
	loss = loss_function;
};

void manager::train(vector<vector<double>> x_train, vector<vector<double>> y_train)
{
	double ls = 0;
	vector<double> predict;
	vector<double> gradient;

	for (int i = 0; i < x_train.size(); i++)
	{
		predict = md->forward(x_train[i]);
		ls += loss->loss(predict, y_train[i]);
		gradient = loss->gradient(predict, y_train[i]);
		md->backward(gradient);
	};
	ls = ls / x_train.size();
};


void manager::test(vector<vector<double>> x_test, vector<vector<double>> y_test)
{
	double ls = 0;
	vector<double> predict;
	vector<double> gradient;

	for (int i = 0; i < x_test.size(); i++)
	{
		predict = md->forward(x_test[i]);
		ls += loss->loss(predict, y_test[i]);
	};
	ls = ls / x_test.size();
};

void manager::fit(vector<vector<double>> x_train, vector<vector<double>> y_train, vector<vector<double>> x_test, vector<vector<double>> y_test, int epoch)
{
	for (int i = 0; i <= epoch;i++)
	{
		std::cout << "Epoch: " << i << "\n";
		train(x_train, y_train);
		test(x_test, y_test);
	};
};