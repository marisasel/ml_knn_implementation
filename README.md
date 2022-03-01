<h1>Implementação do algoritmo k-NN para classificação</h1>

Implementação do algoritmo k-NN(K-Nearest Neighbors) ou k-vizinhos mais próximos, em Python, para classificação. Entrada das bases de treino e de teste no formato SVM Light. O algoritmo k-NN é um método de aprendizado supervisionado não-paramétrico. Este projeto traz uma implementação "manual" do KNN, computacionalmente mais custosa, e uma implementação utilizando funções da biblioteca scikit-learn, que usa árvores de particionamento espacial para particionar o espaço de busca e outros recursos - o que torna seu desempenho em relação ao tempo de execução bem melhor (para k = 3, speed up médio de 6,8 nos testes realizados com as bases disponíveis neste repositório). Em ambas as versões do algoritmo, utilizou-se a distância euclidiana.

## Versão do Python e bibliotecas utilizadas

:warning: [Python 3.8.8](https://www.python.org/downloads/release/python-388/)
:warning: [sys](https://docs.python.org/3/library/sys.html)
:warning: [NumPy](https://numpy.org/)
:warning: [statistics](https://docs.python.org/3/library/statistics.html/)
:warning: [scikit-learn](https://scikit-learn.org/stable/)

## Como rodar a versão implementada manualmente do k-NN :arrow_forward:

No terminal, execute: 

```
python3 knn.py <data_train> <data_test> <k_value>
```

## Como rodar a versão que utiliza funções da biblioteca scikit learn para implementar o k-NN :arrow_forward:

No terminal, execute: 

```
python3 knn_sklearn.py <data_train> <data_test> <k_value>
```
## Entrada:
Ambas as versões do programa devem receber um arquivo com a base de treinamento, um arquivo com a base de testes e o valor de k, um número inteiro que vai determinar o tamanho da vizinhança observada. Os dois arquivos de entrada devem estar no formato SVM Light:

<rótulo da classe> <índice da característica>:<valor da característicca> ... <índice da característica>:<valor da característica>

Nas bases utilizadas para validar a implementação, os vetores de atributos contêm 132 características.

## Saída:
Ambas as versões do programa fornecem como saída a acurácia e a matriz de confusão, que são impressas no terminal. 
