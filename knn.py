#******************  KNN  *******************
#	Implementation:	Marisa Sel Franco   *
#	IBM - UFPR   			    *
#********************************************

#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
import statistics
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def main(data_train, data_test, k):

	# carrega os dados
	print("Carregando dados de treino...")
	X_train, y_train = load_svmlight_file(data_train)
	print("Carregando dados de teste...")
	X_test, y_test = load_svmlight_file(data_test)

	# converte os dados das features para uma matriz
	X_train = X_train.toarray()
	X_test = X_test.toarray()

	# converte as classes de float para int
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)

	# faz a normalização dos dados com scikit-learn
	scaler = preprocessing.MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

	print("Fazendo as predições com o algoritmo k-NN...")
	y_pred = []
	# calcula k-NN e faz as predições
	for test_entry in X_test:
		# calcula as distâncias euclidianas entre uma entrada de teste e todas de treino
		distances = np.linalg.norm(X_train - test_entry, axis=1)
		# ordena o vetor de distâncias e obtém os IDs das k mais próximas
		nearest_neighbor_ids = distances.argsort()[:k]
		nearest_neighbor_labels = y_train[nearest_neighbor_ids]
		# usa a moda para verificar a qual classe atribui a entrada de teste
		label = statistics.mode(nearest_neighbor_labels)
		# incrementa o vetor com a nova predição
		y_pred.append(label)

	# calcula a acurácia
	accuracy = accuracy_score(y_test, y_pred)
	print("Acurácia da predição: ", accuracy)
	
	# cria e imprime a matriz de confusão
	print("Matriz de confusão:\n")
	c_matrix = confusion_matrix(y_test, y_pred)
	print(c_matrix)

if __name__ == "__main__":
	if len(sys.argv) != 4:
		sys.exit("Use: knn.py <data_train> <data_test> <k_value>")

	main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
