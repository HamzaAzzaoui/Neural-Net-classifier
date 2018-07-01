import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

""" Layer Object
"""
class Layer:
    def __init__ (self , input_size , numNeurons ):
        self.input_size = input_size
        self.numNeurons = numNeurons
        self.weights = np.random.randn(input_size , numNeurons)
        self.bias = np.random.randn(numNeurons)
        self.outPut = None



    def matmul(self , previousLayer , weights):
        return np.dot(previousLayer , weights) 

    def forwardPass(self , previousLayer , dropout = .8 , train = False ):
        if train == False:
            return self.matmul(previousLayer , self.weights  )
        if previousLayer.shape[1] != self.input_size:
            print("Error input size doesn't match layer size")
            exit()
        # If train == True use dropOut
        if train:
            
            a = np.random.binomial([np.ones((self.input_size,self.numNeurons))],1-dropout)[0] * (1.0/(1-dropout))
            self.output = self.matmul(previousLayer , self.weights * a )
            return   self.matmul( previousLayer , self.weights * a ) 
        

    def getInputGradient(self , outputGradient):
       
    
        inPutGradient = np.dot(outputGradient , self.weights.T)
        return inPutGradient

class Network:
    def __init__ (self , input_size , num_layers , neuronsPerLayer , dropout = .8):

        if num_layers != len(neuronsPerLayer):
            print("not enough inputs , Check number of layers or  number of neurons per layer")
            exit()

        if num_layers < 2:
            print("Please select at least 2 layers")
            exit()

        if neuronsPerLayer[-1] != 1:
            print("Your output Layer should have only one neuron")
            exit()
        self.J = []
        self.J_val = []
        self.input_size = input_size
        self.num_layers = num_layers
        self.neuronsPerLayer = neuronsPerLayer
        self.Layers = []
        self.dropout = dropout
        self.populated = False


    def softmax(self , x):
          
        if len(x) == 3:
            
            return np.exp(x) / np.sum(np.exp(x))

        sol = np.exp(x) / np.sum(np.exp(x), axis =1 ,keepdims=True)
        return sol

    
    def populateLayer(self):
        if self.populated:
            return
        self.Layers.append(Layer(self.input_size , self.neuronsPerLayer[0]))
        for i in range(1 , self.num_layers):
            l = Layer(self.neuronsPerLayer[i-1] , self.neuronsPerLayer[i])
            self.Layers.append(l)

        self.populated = True
    
    def forwardPass (self , X , train = False):
        output = X
        outputList = [X]
        self.populateLayer()
        activationList = []
        for layer in self.Layers:
            if train:
                output = layer.forwardPass(output ,layer.weights , train)
            else:
                output = np.dot( output , layer.weights)
            #output = layer.forwardPass(output ,layer.weights)
            activationList.append(output)
            output = self.sigmoid(output)
            outputList.append(output)
            X = output
            
        self.activationList = activationList
        self.outputList = outputList
        return outputList

    def predict(self, x):
        x = np.array(x)
        prediction = self.forwardProp((x/x.max()) , False) * 2 
        if prediction < 0.5:
            return 0
            
        elif prediction < 1.5:
            return 2
            
        elif prediction < 2.5:
            return 1
            

    """Utility Functions"""

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def forwardProp(self, x , train = False):
        return self.forwardPass(x , train)[-1]

    # Squared loss
    def costFunction(self, x, y):
        y_ = self.forwardProp(x , False)
        J = 0.5 * sum((y-y_)**2) / len(x)
        return J
	
	
	
    def sigmoidDerived(self, z):
        return ((np.exp(-z)) / ((1 + np.exp(-z))**2))
	
    def sigmoidInverse(self,z):
        return np.log(z/(1-z))


    def costFunctionDerived(self, X, y):
        self.y_ = self.forwardProp(X)   
        
        dJ = []
    
        delta = np.multiply(-(y-self.y_), self.sigmoidDerived(self.activationList[-1]))
        dJi = np.dot(self.outputList[-2].T, delta)
        dJ.append(dJi)
            
        for i in reversed(range(1 ,len(self.Layers))):
        
            delta = np.dot(delta, self.Layers[i].weights.T)*self.sigmoidDerived(self.activationList[i-1])
            dJi = np.dot(self.outputList[i-1].T, delta)  
            dJ.append(dJi)
        
        return dJ
  
    """Utility functions for optimization"""

    def getParams(self):
        params = self.Layers[0].weights.ravel()
        for layer in self.Layers[1:]:
            params = np.concatenate((params, layer.weights.ravel()))
        return params

    def setParams(self, params):
        start = 0
        rows = self.input_size
        cols = self.neuronsPerLayer[0]
        end = rows * cols
        for i in range(len(self.Layers) -1):
            self.Layers[i].weights = np.reshape(params[start:end], (rows , cols))
            start = end 
            rows = cols
            cols = self.neuronsPerLayer[i +1]
            end = start + rows * cols
        self.Layers[len(self.Layers) - 1].weights = np.reshape(params[start:end], (rows , cols))

    
    def computeGradients(self, X, y):
        J = self.costFunctionDerived(X, y)
        
        gradients = J[-1].ravel()
        
        for dJ in reversed(J[:-1]):
            gradients = np.concatenate((gradients , dJ.ravel()))
        return gradients
        
        
    
    def callbackf(self, params):
        self.setParams(params) 
        self.J.append(self.costFunction(self.X, self.Y)) 
        self.J_val.append(self.costFunction(self.X_val, self.Y_val)) 
        #print("DEBUG --")

    def costFunctionWrapper(self, params, X, y):
        self.setParams(params)
        return self.costFunction(X, y), self.computeGradients(X, y)
        
        
    """

    Optimized Training Procedure using scipy optimize function

    """

    def trainOptimized(self, X, y , X_val , Y_val):
        #Create variables for local use
        if self.populated == False:
            self.populateLayer()
        self.X = X
        self.Y = y    
        self.X_val = X_val
        self.Y_val = Y_val   
        params0 = self.getParams() 
        options = {'maxiter': 3500, 'disp' : False} 
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackf) #And optimize
        self.setParams(_res.x)


""" Classical Training Procedure : This contains a "Classical implementation of BackProp" 
    """
    # def trainingStep(self , X , Y , lr = 0.1):
    #     outputList = self.forwardPass(X , True) 
    #     outPutGradient = 0
    #     count = 0
    #     for layer in reversed(self.Layers):

    #     output = outputList.pop()
    #     flag = True
    #     if flag :
    #         flag = False
    #         delta = (output - Y) / Y.shape[0] * self.sigmoidDerived(output)
            
    #         inputGrad = layer.getInputGradient(delta)

    #     else:
    #         delta = outPutGradient  * self.sigmoidDerived(output)
    #         inputGrad = layer.getInputGradient(delta)


    #     LayerInput = outputList[-1]


    #     LayerDelta = np.dot(LayerInput.T , delta)

    #     layer.weights -= lr * LayerDelta

    #     count += 1
    #     outPutGradient = inputGrad
            

    # def train(self , X , Y , num_iter = 20000 , lr = 0.1):
    #     self.populateLayer()
    #     errorLogs = []
    #     for k in range(num_iter):
            
    #         error = self.costFunction(X, Y)
    #         errorLogs.append(error)
    #         self.trainingStep(X , Y , lr )
    #     plt.plot(errorLogs)
    #     plt.ylabel("Error Loss")~
    #     plt.show()
        


    