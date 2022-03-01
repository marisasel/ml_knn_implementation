#************* KNN scikit-learn *************
#	Implementation:	Marisa Sel Franco   *
#	IBM - UFPR			    *
#********************************************

#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

def main(data_train, data_test, k):

	# carrega os dados
	print("Carregando dados de treino...")
	X_train, y_train = load_svmlight_file(data_train)
	print("Carregando dados de teste...")
	X_test, y_test = load_svmlight_file(data_test)

	# converte os dados para um vetor
	X_train = X_train.toarray()
	X_test = X_test.toarray()

	# faz a normalização dos dados
	scaler = preprocessing.MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

	# cria um kNN
	neigh = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean', algorithm = 'auto')
	print('Ajustando kNN...')
	neigh.fit(X_train, y_train)

	# faz a predição do classificador
	print('Predizendo...')
	y_pred = neigh.predict(X_test)

	# mostra o resultado do classificador na base de teste
	print('Acurácia: ',  neigh.score(X_test, y_test))

	# cria a matriz de confusão
	c_matrix = confusion_matrix(y_test, y_pred)
	print(c_matrix)

if __name__ == "__main__":
	if len(sys.argv) != 4:
		sys.exit("Use: knn.py <data_train> <data_test> <k_value>")

	main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
