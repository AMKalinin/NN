﻿// NN.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include "Neuron.h"
#include "Layer.h"
#include "Model.h"

int main()
{
    sigmoida sigma;
    int a = 10;
    input_layer l(2);
    hide_layer hide(2, &sigma);
    output_layer out(2,&sigma);

    nn mod;

    mod.add_layers(&l);
    mod.add_layers(&hide);
    mod.add_layers(&out);

    mod.compile();

    hide.neurons[0]->sum = 0.3;
    hide.neurons[1]->sum = -15;
    hide.activate();
    vector<double> der = hide.derivative();
    std::cout << der[0];
    std::cout << der[1];

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