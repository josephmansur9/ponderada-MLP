import numpy as np

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    # para o valor de log não dar -inf, o minimo é limitado com um valor pequeno (1e-12)
    log_probs = np.log(np.clip(y_pred, 1e-12, 1.0))
    # vai linha por linha, pega os valores logaritmicos das classes verdadeiras e soma tudo
    return np.sum(log_probs[np.arange(n), y_true])

def cross_entropy_gradient(y_pred, y_true):
    # pega o número de linhas (amostras) que estão sendo processadas
    n = y_pred.shape[0]
    # cria uma cópia das previsões para não mudar os dados originais
    grad = y_pred.copy()  
    # multiplica o valor da classe verdadeira por 1
    grad[np.arange(n), y_true] *= -1 
    # retorna a matriz com o gradiente calculado para cada amostra
    return grad