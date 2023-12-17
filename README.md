# Neural Network
## Библиотека предназначена для построения нейронных сетей

**Начало работы:** 
1. Подключить следующие модули к главному файлу:
    ```python
    import nn
    from functions import *
    from layer import *
    from manager import*

2. Cоздать модель (наша сеть в которую мы будем добавлять слои):
    ```python
    model = nn.NeuralNetwork()
3. Построить сеть
   Слои имеют пареметры: тип слоя, и его размер(количество нерйронов)
   Также выходной и скрытые слои имеют функции активации(пока доступны только sigmoida, RELU, tanh и Softmax для выходного слоя). Если функция активации не нужна, то в параметр слоя подаётся функция `nonFunc()` 
    ```python
    model.add_layers(input_layer('input', 4))
    model.add_layers(hide_layer('hide', 10,sigmoida()))
    model.add_layers(output_layer('output', 3, softmax()))
4. Скомпилировать модель(создать связи между нейроннами и первоначально проинициализировать веса)
    ```python 
    model.compile()

5. Подать модель 'менеджеру'. Также вырать *learning rate*, функцию потерь (SSE - сумма квадратов ошибки, SoftMaxCrossEntropy - перекрёстная энропия)
    ```python 
    mg = manager(mщвудd, 0.01, SSE())

6. Запустить обучение
    ```python
    mg.fit(x_train,y_train,x_test, y_test, 100)

    

    