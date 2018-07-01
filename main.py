from neuron import *
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np
"""Utility function to split the dataset"""
def load_data(split):
    iris = datasets.load_iris()
    
    x = iris.data
    y = iris.target
    x = x/x.max()
    y = y/y.max()
    y = np.reshape(y, (-1,1))  

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split) 
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
     
    return x_train , y_train , x_test , y_test , x_val , y_val

"""Utility function to match each class with an output"""
def convertGrTruth(y):
    if y == 0:
        return 0
    elif y == 0.5:
        return 2
    elif y == 1:
        return 1

if __name__ == "__main__":
    
    
    import argparse

    parser = argparse.ArgumentParser(description='Train IRIS Dataset')
    parser.add_argument('--NumLayers', type=int, default=2,
                        help=" Int :Numbero of layers in the network")

    parser.add_argument('--NodePerLayer', type=list, default=[4 , 1],
                        help=' List :Number of neurons per Layer.')

    parser.add_argument('--SplitRatio', type=float, default=0.3,
                        help='Float :Data set split ratio')
    
    args = parser.parse_args()

    X_train , Y_train , X_test , Y_test , X_val , Y_val = load_data(args.SplitRatio)
    net = Network(4 , args.NumLayers , args.NodePerLayer , dropout = 1)
    
    net.trainOptimized(X_train , Y_train , X_val , Y_val)
    

    plt.plot(net.J , 'green')
    plt.plot(net.J_val , 'blue')
    plt.grid(1)
    plt.title("Training Loss in Green VS Validation Loss in Blue")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    count = 0
    divi = len(X_test)
    for i in range(len(X_test)):
        if convertGrTruth(Y_train[i][0]) == net.predict(X_train[i]):
            count += 1
    count = count * 100
    print("---------------------------")
    print("TEST ACCURACY IS " ,"%.2f" % (count / divi) , "%")
    print("---------------------------")
    