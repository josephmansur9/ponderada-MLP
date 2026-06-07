import numpy as np

def relu(z):
    # Aplica a ativação ReLU retornando zero para valores negativos e o próprio valor para positivos.
    return np.maximum(0, z)

def relu_derivative(z):
    # Calcula a derivada da ReLU, que assume valor 1.0 se z > 0 e 0.0 se não foi maior que 0.
    return (z > 0).astype(float) # Converte o array booleano (True/False) em números (1.0/0.0).

def softmax(z):
    #Calcula o Softmax tradicional que faz com que a soma das probabilidades seja 1
    exp_z = np.exp(z) # Eleva o número de Euler 'e' à potência de cada elemento de z.
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) # Normaliza dividindo cada elemento pela soma de sua linha.

# faz um print de teste pra ver se o consegue lidar com os números
print(softmax(np.array([[1000.0, 2000.0]])))