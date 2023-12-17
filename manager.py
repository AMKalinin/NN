from tqdm import tqdm

class manager:
    def __init__(self, model, learning_rate, loss_function):
        self.model = model
        self.model.lr = learning_rate
        self.loss = loss_function
        self.loss_train_list = []
        self.loss_test_list = []
        self.accuracy_train = []
        self.accuracy_test = []
        
    def train(self,x_train, y_train):
        loss = 0
        correct = 0
        for i in tqdm(range(len(x_train)), desc='train', leave=False):
            predict = self.model.forward(x_train[i])
            correct += self.accuracy(predict, y_train[i])
            loss += self.loss.loss(predict, y_train[i])
            gradient = self.loss.gradient(predict, y_train[i])
            self.model.backward(gradient)
        total_loss = loss/len(x_train)
        acc = correct/len(x_train)
        self.loss_train_list.append(total_loss)
        self.accuracy_train.append(acc)
        print('train loss:    ', total_loss)
        print('      accuracy:', acc)
            
    def test(self,x_test, y_test):
        loss = 0
        correct = 0
        for i in tqdm(range(len(x_test)), desc='test', leave=False):
            predict = self.model.forward(x_test[i])
            correct += self.accuracy(predict, y_test[i])
            loss += self.loss.loss(predict, y_test[i])
        total_loss = loss/len(x_test)
        self.loss_test_list.append(total_loss)
        acc = correct/len(x_test)
        self.accuracy_test.append(acc)
        print('test loss:    ', total_loss)
        print('     accuracy:', acc)
        print()

    def accuracy(self,predict, target):
        max_ind1 = 0
        max_ind2 = 0
        max1 = predict[0]
        max2 = target[0]
        for i in range(1,len(predict)):
            if max1 < predict[i]:
                max1 = predict[i]
                max_ind1 = i
            if max2 < target[i]:
                max2 = target[i]
                max_ind2 = i
        if max_ind1 == max_ind2:
            return 1
        else:
            return 0 
        

    def fit(self, x_train,y_train,x_test,y_test, epoch):
        for i in range(1, epoch+1):
            print('Epoch: ', i, '/', epoch)
            self.train(x_train,y_train)
            self.test(x_test,y_test)

