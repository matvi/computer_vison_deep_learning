
import math
import numpy as np

class perceptron:

    # this script calculates creates a simple neural network
    # capable of learning to calculate the XOR Gate.
    # This example is used to demostrait how backpropagation works
    # This example uses batch gradient descent to finde the local minimum
        #The results are written below
    #   x0 = 1 , x1 = 1  y0 =  0.0668454342501834
    #   x0 = 1 , x1 = 0  y0 =  0.9538048625755944
    #   x0 = 0 , x1 = 1  y0 =  0.9538040120861836
    #   x0 = 0 , x1 = 0  y0 =  0.03565435591075458
    #   error ->  0.0050038025982497725
    #   epoach ->  1748000

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


    def xor(self):
        print('xor')
        X = np.array([[1,1],[1,0],[0,1],[0,0]]) #X.shape = (4,2)
        y = np.array([0,1,1,0])
        w0 = np.array([[.9,.1],[.3,.5]])
        w1 = np.array([.8,.7])
        epoachs = 0
        error = 1
        max_epoach = 100000
        accepted_error= .005

        while error > accepted_error or epoachs < max_epoach:
            #forward pass
            youtput=[]
            epoachs += 1
            for i in range(X.shape[0]):#X.shape = (4,2)
                h0 = self.sig(w0[0,0]*X[i][0] + w0[1,0]*X[i][1])
                h1 = self.sig(w0[0,1]*X[i][0] + w0[1,1]* X[i][1])
                y0 = self.sig(w1[0]* h0 + w1[1] * h1) # shape = (4,)
                youtput.append(y0)
                if epoachs % 1000 ==0:
                    print('x0 =', X[i][0], ', x1 =', X[i][1], ' y0 = ',y0)

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

            error = self.error(y,youtput )
            if epoachs % 1000 == 0:
                print('error -> ', error)
                print('epoach -> ', epoachs)  

    #doesnt work
    def xorGeneralized(self):
        print('xor')
        X = np.array([[1,1],[1,0],[0,1],[0,0]]) #X.shape = (4,2)
        y = np.array([0,1,1,0])
        w0 = np.array([[.9,.1],[.3,.5]])
        w1 = np.array([.8,.7])
        epoachs = 0
        error = 1
        max_epoach = 100000
        accepted_error= .005

        while error > accepted_error or epoachs < max_epoach:
            #forward pass
            y_output= np.zeros([y.shape[0]]) #initializing the output -> np.zeros([4])
            epoachs += 1
            h= np.zeros([2]) # two outputs
            for l in range(4):
                for j in range(h.shape[0]):    
                    for i in range(X.shape[1]):#X.shape = (4,2)
                        h[j] = self.sig(w0[j,i]*X[l][i])
                for j in range(1): #y.shape[1]= 1    
                    for i in range(h.shape[0]):#X.shape = (4,2)
                       # print('l ', l, ' i ', i, ' j, ', j)
                        y_output[l] = self.sig(w1[j]* h[j] + w1[j+1] * h[j+1])
            #print (y_output)
            #error = .00003
            #epoachs = 100000000 
                #h0 = self.sig(w0[0,0]*X[i][0] + w0[1,0]*X[i][1])
                #h1 = self.sig(w0[0,1]*X[i][0] + w0[1,1]* X[i][1])
            #for i in range(y.shape[0]):
               # y_output[i]= self.sig(w1[0]* h[i] + w1[i+1] * h[i+1]) # shape = (4,)
                #y0 = self.sig(w1[0]* h[0] + w1[1] * h[1]) # shape = (4,)
                #youtput.append(y0)
                #if epoachs % 1000 ==0:
                #    print('x0 =', X[i][0], ', x1 =', X[i][1], ' y0 = ',y0)

                #backpropagation
                dey0 = -(y[i]-y_output[l]) # y[i] -> desired output | y0 -> output
                lay1 = dey0 * y_output[l] * (1 - y_output[l])
                deW0_00 = lay1 * w1[0] * h[0] * (1 - h[0]) * X[l][0]
                deW0_01 = lay1 * w1[1] * h[1] * (1 - h[1]) * X[l][0]
                deW0_10 = lay1 * w1[0] * h[0] * (1 - h[0]) * X[l][1]
                deW0_11 = lay1 * w1[1] * h[1] * (1 - h[1]) * X[l][1]
                deW1_00 = dey0 * h[0]
                deW1_10 = dey0 * h[1]
                w0[0,0] = self.gradient(w0[0,0], deW0_00)
                w0[0,1] = self.gradient(w0[0,1], deW0_01)
                w0[1,0] = self.gradient(w0[1,0], deW0_10)
                w0[1,1] = self.gradient(w0[1,1], deW0_11)
                w1[0] = self.gradient(w1[0], deW1_00)
                w1[1] = self.gradient(w1[1], deW1_10)

            #error = self.error(y,y_output)
            if epoachs % 1000 == 0:
                print('error -> ', error)
                print('epoach -> ', epoachs) 
                print('expected,', y)
                print('output,', y_output)

    def gradient(self, w, w_derivative):
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

