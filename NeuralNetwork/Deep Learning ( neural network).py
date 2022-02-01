#!/usr/bin/env python
# coding: utf-8

# # Manual Neural Network

# In[1]:


from numpy import exp,array,random,dot

class MyNeuralNetwork:
    def __init__(self):
        random.seed(1)   # for output --the  intial input
        layer2=5
        layer3=4
        self.weights1=2*random.random((3,5))-1 #((-1) to train NN properly)
        self.weights2=2*random.random((5,4))-1
        self.weights3=2*random.random((4,1))-1
        
    def mySigmoid(self,x):
        return 1/(1+exp(-x))
    
    def mySigmoidDerivative(self,x):
        return x*(1-x)
    
    def myTrain(self,inputs,expected_outputs,iterations):
        for i in range (iterations):
            
            dota2=dot(inputs,self.weights1)
            a2=self.mySigmoid(dota2)
            
            dota3=dot(a2,self.weights2)
            a3=self.mySigmoid(dota3)
            
            outputdot=dot(a3,self.weights3)
            actual_output=self.mySigmoid(outputdot)
            
            del4=(expected_outputs-actual_output)*self.mySigmoidDerivative(actual_output)
            
            del3=dot(self.weights3,del4.T)*self.mySigmoidDerivative(a3).T
            
            del2=dot(self.weights2,del3)*self.mySigmoidDerivative(a2).T
            
            adjustment3=dot(a3.T,del4)
            adjustment2=dot(a2.T,del3.T)
            adjustment1=dot(inputs.T,del2.T)
            
            self.weights1=self.weights1+adjustment1
            self.weights2=self.weights2+adjustment2
            self.weights3=self.weights3+adjustment3
            
    def forwardPass(self,inputs):
        a2=self.mySigmoid(dot(inputs,self.weights1))
        a3=self.mySigmoid(dot(a2,self.weights2))
        output=self.mySigmoid(dot(a3,self.weights3))
        
        return output
    
    
    
nn=MyNeuralNetwork()

ip=array([[0,0,1],[1,1,1],[1,0,1],[1,1,0]])
op=array([[0,1,1,0]]).T
nn.myTrain(ip,op,2000)
res=nn.forwardPass(array([1,1,1]))
print(res)
            
    
            
            
            
            
            
        
        
        
    


# In[ ]:




