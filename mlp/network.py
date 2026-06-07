import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy_loss, cross_entropy_gradient

class MLP:
    # MLP consiste de multiplas camadas de neurônios totalmente conectados, onde cada neurônio em uma camada está conectado a todos os neurônios da camada seguinte. Ele consegue de aprender representações complexas dos dados, o que faz ele bom para classificação por exemplo.
    def __init__(self, layer_sizes, learning_rate=0.01):
        # self.lr é a taxa de aprendizado que controla o tamanho de passos que damos na direção do gradiente para minimizar a função de perda. Como foi explicado no outro arquivo um valor muito alto ou muito baixo pode levar a divergência ou a um treinamento muito lento, respectivamente.
        self.lr = learning_rate
        #self weights e self.biases são listas que armazenam os pesos e vieses de cada camada da rede. Os pesos são inicializados usando a técnica de He, pensanso na função de ativação ReLU, enquanto os vieses são inicializados como zeros.
        self.weights = []
        self.biases = []

        # esse loop percorre as camadas da rede, calculando o número de entradas (fan_in) para cada camada e inicializando os pesos e vieses de acordo.
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # Inicialização He para os pesos e inicialização nula para os vieses, onde os pesos são multiplicados por sqrt(2.0 / fan_in) para manter a variância das ativações constante ao longo das camadas, o que ajuda a evitar problemas de vanishing ou exploding gradients.
            W = np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            # Inicialização nula para os vieses, onde cada viés é inicializado como zero, o que é uma prática comum para evitar que os neurônios comecem com uma ativação diferente, o que pode ajudar a rede a aprender de forma mais eficiente.
            b = np.zeros((1, layer_sizes[i+1]))
            # os pesos e vieses são adicionados às listas self.weights e self.biases, respectivamente, para que possam ser utilizados posteriormente durante o processo de treinamento da rede.
            self.weights.append(W)
            self.biases.append(b)


    # essa função forward pass calcula a predição da rede camada por camada, é guardado tudo no cache porque o backward vai precisar desses valores para calcular os gradientes via regra da cadeia
    def forward(self, X):
        # cache['A'] guarda as ativações de cada camada, A[0] é o próprio input; A[1] é após a primeira camada, e assim por diante.
        cache = {'A': [X]}
        A = X

        # camadas ocultas usam ReLU como ativação, i vai de 0 até a penúltima camada (o -1)
        for i in range(len(self.weights) - 1):
            # Z é a combinação linear: entrada × pesos + bias, onde A é a ativação da camada anterior, self.weights[i] são os pesos da camada atual e self.biases[i] são os vieses da camada atual. O resultado Z é então passado pela função de ativação ReLU para obter a ativação A da camada atual.
            Z = A @ self.weights[i] + self.biases[i]
            # aplicamos ReLU: neurônios com Z <= 0 não ativam ja que A = 0
            A = relu(Z)
            # guardamos Z (antes do ReLU) porque relu_derivative precisa de Z, porém não de A porque a derivada depende do sinal antes da ativação que é Z
            cache[f'Z{i}'] = Z
            cache['A'].append(A)

        # a última camada usa softmax, não ReLU, softmax transforma os logits em probabilidades que somam 1. Essa é a camada de saída onde cada posição representa a probabiliadde de ser aquele dígito
        Z_out = A @ self.weights[-1] + self.biases[-1]
        y_pred = softmax(Z_out)
        # a ultima ativação é a predição final, então guardamos o Z da última camada para o backward e a predição final para calcular a perda e os gradientes
        cache[f'Z{len(self.weights) - 1}'] = Z_out
        cache['A'].append(y_pred)
        return y_pred, cache

    def backward(self, cache, y_true):
        #o backward pass aplica a regra da cadeia de trás pra frente, assim conseguimos dizer quanto cada peso contribuiu para o erro. O que é basicamente o gradiente, a direção em que devemos mover os pesos pra reduzir a perda.
        n_layers = len(self.weights)
        # gradientes dos pesos e dos biases são inicializados como listas de None, que serão preenchidas com os valores calculados durante o processo de backpropagation. dWs[i] armazenará o gradiente dos pesos da camada i, enquanto dbs[i] armazenará o gradiente dos vieses da camada i.
        dWs, dbs = [None] * n_layers, [None] * n_layers
        #Nesse ponto vemos o gradiente da loss em relação à saída da rede.
        # dA aqui representa d(Loss)/d(A), o quanto a loss muda com a ativação
        dA = cross_entropy_gradient(cache['A'][-1], y_true)

        #percorre o loop de tras pra frente 
        for i in reversed(range(n_layers)):
            # A_prev é a ativação da camada anterior
            A_prev = cache['A'][i]
            # Z é a combinação linear antes da ativação, que é necessária para calcular a derivada da ReLU. Ele é usado para determinar quais neurônios estavam ativos (Z > 0) e quais não estavam (Z <= 0), o que é importante para calcular o gradiente corretamente.
            Z = cache[f'Z{i}']

            # na ultima camada, dA ja é o gradiente combinado, então não aplicamos a ReLU porque usamos softmax. E essa derivada do softmax ja foi utiliza da na função cross_entropy_gradient, então dA já é o gradiente correto para a última camada.
            if i == n_layers - 1:
                dZ = dA
            else:
                # aplica o produto matricial depois da regra da cadeia, onde cada elemento de dA é multiplicado pela derivada da ReLU
                dZ = dA @ relu_derivative(Z)

            # essa parte descobre a culpa dos pesos multiplicando o sinal que entrou (A_prev) pelo erro que saiu (dZ), se a entrada foi grande e o erro foi grande, o peso precisa mudar muito
            dWs[i] = A_prev.T @ dZ

            # essa parte descobre a culpa dos vieses (biases). Como eles não multiplicam nenhuma entrada, a culpa deles é o próprio erro (dZ) somamos o erro de todo o batch (axis=0)
            dbs[i] = np.sum(dZ, axis=0, keepdims=True)

            # Por fim a culpa é passadapara a camada de trás, ela joga o erro atual (dZ) de volta pelos pesos (W), esse dA será o ponto de partida para a camada anterior calcular seus próprios erros
            dA = dZ @ self.weights[i].T

        return dWs, dbs