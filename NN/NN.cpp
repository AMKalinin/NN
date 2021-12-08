// NN.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include "Neuron.h"
#include "Layer.h"
#include "Model.h"

int main()
{
    sigmoida sigma;
    int a = 10;
    input_layer l(3);
    output_layer out(1,&sigma);

    nn mod;

    mod.add_layers(&l);
    mod.add_layers(&out);

    mod.compile();

    mod.layers[1]->neurons[0]->w[0] = -0.16595599;
    mod.layers[1]->neurons[0]->w[1] = 0.44064899;
    mod.layers[1]->neurons[0]->w[2] = -0.99977125;

    /*mod.layers[1]->neurons[0]->w[0] = 9.67299303;
    mod.layers[1]->neurons[0]->w[1] = -0.2078435;
    mod.layers[1]->neurons[0]->w[2] = -4.62963669;*/

    vector<double> x1 = { 0, 0, 1 };
    vector<double> x2 = { 1, 1, 1 };
    vector<double> x3 = { 1, 0, 1 };
    vector<double> x4 = { 0, 1, 1 };
    vector<double> x5 = { 1, 0, 0 };

    vector<double> y1 = {0};
    vector<double> y2 = {1};
    vector<double> y3 = {1};
    vector<double> y4 = {0};


    vector<double> grad = { 0 };

    vector<double> predict;

    for (int i = 0; i < 10000; i++)
    {
        std::cout << i << "\n";
        predict = mod.forward(x1);
        grad[0] = 2 * ( predict[0]-y1[0]);
        mod.backward(grad);

        predict = mod.forward(x2);
        grad[0] = 2 * (predict[0] - y2[0]);
        mod.backward(grad);

        predict = mod.forward(x3);
        grad[0] = 2 * (predict[0] - y3[0]);
        mod.backward(grad);

        predict = mod.forward(x4);
        grad[0] = 2 * (predict[0] - y4[0]);
        mod.backward(grad);

    };
    
    predict = mod.forward(x5);
    std::cout << predict[0];

}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
