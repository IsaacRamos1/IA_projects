
"""
Created on Sun Nov 10 14:19:32 2019

@author: isac_
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

### utlização do banco de dados iris:
### https://archive.ics.uci.edu/ml/datasets/Iris
###
### inicialização das variaveis
iris = datasets.load_iris() 
entradas = iris.data
saidas = iris.target

### configurando a rede neural para treinamento
redeNeural = MLPClassifier(verbose = True,
                           max_iter = 10000,
                           tol = 0.0000001,
                           activation = 'logistic',
                           learning_rate_init=0.0003)

### iniciando treinamento
redeNeural.fit(entradas, saidas)

### predição de valores
redeNeural.predict([[2, 4, 5.5, 1], [1, 3.5, 4.5, 5]])
#valores quaisquer dentro de uma faixa de aceitação


