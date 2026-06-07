import numpy as np

def relu(z):
    # Aplica a ativação ReLU retornando zero para valores negativos e o próprio valor para positivos.
    return np.maximum(0, z)

def relu_derivative(z):
    # Calcula a derivada da ReLU, que assume valor 1.0 se z > 0 e 0.0 se não foi maior que 0.
    return (z > 0).astype(float) # Converte o array booleano (True/False) em números (1.0/0.0).

def softmax(z):
    # calcula o softmax subtraindo o maximo de cada linha antes de aplicar a exponencial para não ter overflow, no teste passado recebi nan e nan
    z_stable = z - np.max(z, axis=1, keepdims=True)
    # eleva os valores a exponencial 
    exp_z = np.exp(z_stable)
    # normaliza as exponenciais para que cada linha seja uma distribuição de probabilidade (soma 1)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# faz um print de teste pra ver se o consegue lidar com os números
print(softmax(np.array([[1000.0, 2000.0]])))