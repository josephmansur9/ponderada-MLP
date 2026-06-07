class SGD:
    # o SGD é uma técnica de otimização que atualiza os pesos e vieses da rede neural usando o gradiente descendente estocástico. Ou seja no lugar de calcular o erro de todos os dados, ele seleciona uma amostra aleatória para cada iteração``
    def __init__(self, learning_rate=0.01):
        # learning_rate é a taxa de aprendizado, que controla o tamanho dos passos que damos na direção do gradiente para minimizar a função de perda. Um valor muito alto pode fazer com que o modelo acabe divergindo muito, enquanto um valor muito baixo pode resultar em um treinamento muito lento.
        self.lr = learning_rate
        # o método update é responsável por atualizar os pesos e vieses da rede neural com base nos gradientes calculados anteriormente. Ele recebe as listas de pesos, vieses, gradientes dos pesos (dWs) e gradientes dos vieses (dbs) e atualiza cada um deles subtraindo o produto do gradiente pela taxa de aprendizado.
    def update(self, weights, biases, dWs, dbs):
        # Passa por cada camada da rede usando o índice 'i'
        for i in range(len(weights)):
            # Subtrai o gradiente diretamente do peso
            weights[i] -= dWs[i]
            
            # Subtrai o gradiente diretamente do viés
            biases[i] -= dbs[i]