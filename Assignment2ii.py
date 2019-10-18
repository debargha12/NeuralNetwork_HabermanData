#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection


# In[2]:


class NeuralNet:
    
    def __init__(self, train,df, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        train_dataset = self.preprocess(train,"train",df)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #


    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        elif activation == "ReLu":
            self.__ReLu(self, x)

    #

    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "ReLu":
            self.__ReLu_derivative(self, x)
            
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __tanh(self, x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def __ReLu(self, x):
        return np.maximum(x, 0)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def __tanh_derivative(self, x):
        return (1-(x*x))
    def __ReLu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    #

    #

    def preprocess(self, X,t_o_t,df):
        if t_o_t=="train":
            
            df1 = X.iloc[:, 3:]
            dfx=df.iloc[:,:3]
            ls=df1.values.tolist()
            nps=np.asarray(ls)
            npsn= nps-1

            df2=X.iloc[:,:3]

            df_norm = (df2 - dfx.min()) / (dfx.max() - dfx.min())

            df_norm['y']=npsn
            
            return df_norm
        
        else:
            
            df3 = X.iloc[:, 3:]
            dfx=df.iloc[:,:3]
            ls1=df3.values.tolist()
            nps1=np.asarray(ls1)
            npsn1= nps1-1

            df4=X.iloc[:,:3]

            df_norm1 = (df4 - dfx.min()) / (dfx.max() - dfx.min())

            df_norm1['y']=npsn1
            return df_norm1
            
        
        
        
        
        

    # Below is the training function

    def train(self,activation, max_iterations = 1000, learning_rate = 0.05):
        
        for iteration in range(max_iterations):            
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
        #f= open("log_assignment2.txt","a+")
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        #f.write("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error))+"\n")
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
            
            


    def forward_pass(self,activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            
            in1 = np.matmul(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            return out
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
            return out
            
        elif activation == "ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__ReLu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__ReLu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__ReLu(in3)
            return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)



    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))
        self.deltaOut = delta_output



    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))
            
        self.delta23 = delta_hidden_layer2



    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1




    def predict(self, test,activation,df, header = True):
        print("--Prediction--")
    


        test_dataset = self.preprocess(test,"test",df)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])
        if activation=="sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            error = 0.5 * np.power((out - self.y), 2)
            print("total error in test dataset",np.sum(error))
        elif activation =="tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
            error = 0.5 * np.power((out - self.y), 2)
            print("total error in test dataset",np.sum(error))
        elif activation=="ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__ReLu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__ReLu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__ReLu(in3)
            error = 0.5 * np.power((out - self.y), 2)
            print("total error in test dataset",np.sum(error))
        


# In[3]:


if __name__ == "__main__":
    functions=["sigmoid","tanh","ReLu"]
    df = pd.read_csv('Large_train.csv')
    trainDF, testDF = model_selection.train_test_split(df, test_size=0.2,shuffle= True, stratify=df["y"])
    #f= open("log_assignment2.txt","a+")
    for activation in functions:
        print("Using: ",activation)
        neural_network = NeuralNet(trainDF,trainDF)
        neural_network.train(activation)
        testError = neural_network.predict(testDF,activation,trainDF)

