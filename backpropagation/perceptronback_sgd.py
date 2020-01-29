
import math
import numpy as np

# this script calculates creates a simple neural network
    # capable of learning to calculate the XOR Gate.
    # This example is used to demostrait how backpropagation works
    # This example uses stochastic gradient descent to finde the local minimum
        #The results are written below
        #   x0 = 1 , x1 = 1  y0 =  0.17432816612083576
        #   error ->  0.015195154751526855
        #   epoach ->  730000
        #   x0 = 1 , x1 = 0  y0 =  0.8771490851648861
        #   error ->  0.007546173637912205
        #   epoach ->  730000
        #   x0 = 0 , x1 = 1  y0 =  0.8771422067555696
        #   error ->  0.007547018680445599
        #   epoach ->  730000
        #   x0 = 0 , x1 = 0  y0 =  0.10004037948553275
        #   error ->  0.0050040387638047015
        #   epoach ->  730000

    #Results with batch gradient descent
    #   x0 = 1 , x1 = 1  y0 =  0.17432816612083576
    #   x0 = 1 , x1 = 0  y0 =  0.8771490851648861
    #   x0 = 0 , x1 = 1  y0 =  0.8771422067555696
    #   x0 = 0 , x1 = 0  y0 =  0.10004037948553275
    #   error ->  0.035292385833689356
    #   epoach ->  730000
    
    #   x0 = 1 , x1 = 1  y0 =  0.0668454342501834
    #   x0 = 1 , x1 = 0  y0 =  0.9538048625755944
    #   x0 = 0 , x1 = 1  y0 =  0.9538040120861836
    #   x0 = 0 , x1 = 0  y0 =  0.03565435591075458
    #   error ->  0.0050038025982497725
    #   epoach ->  1748000

class perceptron:
    def error(self, y, yhat):
        e = 0
        e = e + .5 * (y- yhat)**2
        return e
    def sig(self,x):
        return 1 / (1 + math.exp(-x)) 
    
    def derivative(x):
        return x * (1 - x)


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

                error = self.error(y[i],y0 )
                if epoachs % 1000 == 0:
                    print('error -> ', error)
                    print('epoach -> ', epoachs)  

   

    def gradient(self, w, w_derivative):
        alpha = .001
        w = w - alpha * w_derivative
        return w
        

p1 = perceptron()
p1.xor()



