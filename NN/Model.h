#pragma once
#include <vector>
#include "Layer.h"
using std::vector;
class layer;

class model
{
public:
	virtual void add_layers(layer* lay) = 0;
	virtual void compile() = 0;
	virtual vector<double> init_w(int size) = 0;
	virtual vector<double> forward(vector<double> X) = 0;
	virtual void backward(vector<double> gradient) = 0;
	double lr = 1;
};

class nn : public model
{
	public:
		vector<layer*> layers;
		nn() {};
		void add_layers(layer * lay);
		void compile();
		vector<double> init_w(int size);
		vector<double> forward(vector<double> X);
		void backward(vector<double> gradient);
		
};
