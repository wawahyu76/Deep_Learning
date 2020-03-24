#Belajar Membuat Neural Networking

#--Modules & Functions
import numpy as np

#RelU Activation
def ReLU(input):
    return np.maximum(0, input)

#Sigmoid
def sigmoid(input):
    return 1/(1 + 2.71828**(-input))

#--Neural Networking Architecture
# 1 Input
# 4 Neuron in H1 (RelU)
# 2 Neuron in H2 (Sigmoid)
# 1 Neuron in Output  (Linear)

#--Total Parameter
# Input - H1 > 4 weight + 4 bias (IH1)
# H1 - H2 > 8 weight + 2 bias (H1H2)
# H2 - Output > 2 weight + 1 bias (H2O)

#--Setting Initial Parameter

#Input And Output
inp = 2.0
out = 3.0

# IH1 weight and bias
W_IH1 = np.array([[0.25, 0.5, 0.75, 1]])
B_IH1 = np.array([[1, 1, 1, 1]])

# H1H2 weight and bias
W_H1H2 = np.array([[1, 0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75]])
B_H1H2 = np.array([1, 1])

# H2O weight and bias
W_H20 = np.array([1, 0.5])
B_H20 = np.array([1])

#--Forward Pass

#INPUT - H1
IH1 = np.dot(inp, W_IH1)
IH1 = IH1 + B_IH1

#Activation Function Test
for i in range(len(IH1)):
    IH1[i] = ReLU(IH1[i])
    
#H1 - H2
H1H2 = np.dot(IH1, W_H1H2) + B_H1H2

#Activation Function
for i in range(len(H1H2)):
    H1H2[i] = sigmoid(H1H2[i])
    
# H2 - Output
H20 = np.dot(H1H2, W_H20) + B_H20

#Loss Function
L = (H20 - out)**2/2
    





