
import math
import numpy as np

class perceptron:
    #def __init__(self):

    #def feedforward(X,y, W):
    #Iterates in each layer    
        #for n in range(size(W[:0])) 
            #Calcualte the entry of each neuron in n layer
         #   if(n==1):
          #      for i in range(size(W[:])):
           #         for j in range (size(X)):
            #            a[n][i] = a[n][i]+ X[j]*W[j][i]
            #else :
             #   for i in range(size(W[:])):
              #      for j in range (size(a[n][i])):
               #         a[n][i] = a[n][j]+ a[n-1][j]*W[j][i]

    def sig(self,x):
        return 1 / (1 + math.exp(-x)) 
    
    def derivative(x):
        return x * (1 - x)
    
    def gradientDescendent(w, alpha, epoach):
        for i in range(epoach):
            w = w - alpha * derivative(w)
        return w
    
    def error(S, y):
        for i in range (size(y)):
            e = e + .5 * (S[i]- y[i])**2
        return e

#The goal of backpropagation is to compute the partial derivatives
    #def dydw(a,e,X):
        #for k in range(size(a)):
            #for j in range(size(a)):
                #if(k>1):
                    #X[j]*a[2][k]*

    def xor(self):
        print('xor')
        X = np.array([[1,1],[1,0],[0,1],[0,0]]) #X.shape = (4,2)
        y = np.array([0,1,1,0])
        w0 = np.array([[.9,.1],[.3,.5]])
        w1 = np.array([.8,.7])
        for h in range(10000):
            #forward pass
            youtput=[]
            for i in range(X.shape[0]):#X.shape = (4,2)
                #print('x0', X[i][0])
                #print('x1', X[i][1])
                h0 = self.sig(w0[0,0]*X[i][0] + w0[1,0]*X[i][1])
                h1 = self.sig(w0[0,1]*X[i][0] + w0[1,1]* X[i][1])
                y0 = self.sig(w1[0]* h0 + w1[1] * h1) # shape = (4,)
                youtput.append(y0)
                print('y0',y0)

                #backpropagation
                dey0 = -(y[i]-y0) # y[i] -> desired output | y0 -> output
                deW0_00 = dey0 * y0 * (1 - y0) * w1[0] * h0 * (1 - h0) * X[i][0]
                deW0_01 = dey0 * y0 * (1 - y0) * w1[1] * h1 * (1 - h1) * X[i][0]
                deW0_10 = dey0 * y0 * (1 - y0) * w1[0] * h0 * (1 - h0) * X[i][1]
                deW0_11 = dey0 * y0 * (1 - y0) * w1[1] * h1 * (1 - h1) * X[i][1]
                deW1_00 = dey0 * h0
                deW1_10 = dey0 * h1
        
                w0[0,0] = self.gradient(w0[0,0], deW0_00)
                w0[0,1] = self.gradient(w0[0,1], deW0_01)
                w0[1,0] = self.gradient(w0[1,0], deW0_10)
                w0[1,1] = self.gradient(w0[1,1], deW0_11)
                w1[0] = self.gradient(w1[0], deW1_00)
                w1[1] = self.gradient(w1[1], deW1_10)
        
            print('error -> ', self.error(y,youtput ))  
            #forward pass
            youtput2= []
            for i in range(X.shape[0]):#X.shape = (4,2)
                print('x0 =', X[i][0], ', x1 =', X[i][1])
                h0 = self.sig(w0[0,0]*X[i][0] + w0[1,0]*X[i][1])
                h1 = self.sig(w0[0,1]*X[i][0] + w0[1,1]* X[i][1])
                y0 = self.sig(w1[0]* h0 + w1[1] * h1)
                youtput2.append(y0)
                print('y0----->',y0)
            print('error -> ', self.error(y,youtput2 ))

    def gradient(self, w, w_derivative):
        alpha = .001
        w = w - alpha * w_derivative
        return w

    def gradients(self, w, w_derivative):
        alpha = .001
        w = w - alpha * w_derivative
        return w
    
    def error(self, y, yhat):
        e = 0
        for i in range (y.shape[0]):
            e = e + .5 * (y[i]- yhat[i])**2
        return e 
        

p1 = perceptron()
p1.xor()
        #for i in range(10000):
