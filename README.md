# previsaoprecoscasapy
Previsão com preços de casas usando modelos da biblioteca scikit-learn. 

## Projeto de Previsão de Preços

### Bibliotecas utilizadas
* NumPy
* pandas
* matplotlib
* scikit-learn

### Modelos utilizados
* Regressão Linear
* Árvore de Decisão
* Gradient Boosting
* Random Forest

### Problema

> Muitas vezes antes de fazer uma compra de algum produto, carro ou imóvel temos que fazer pesquisas e encontrar um preço que seja válido, mas não temos conhecimento sobre o preço que iremos pagar. Com essa solução é possível analisar uma previsão dos preços que serão encontrados com determinadas características fornecidas, assim facilitando a busca.

### Objetivo/Solução

> Fazer previsões de preços com um conjunto de dados CSV. Este conjunto de dados consiste em preços de casas com as seguintes características: localização, área, banheiros e quartos. Nas previsões não será utilizado a localização na influência dos preços.

### Como funciona

> Cada modelo é avaliado por meio do cálculo do erro absoluto médio, também fornecido pela biblioteca scikit-learn, junto aos novos dados que serão utilizados para a previsão do valor com as determinadas características fornecidas. Com base nesses resultados de erro e preço é possível analisar qual modelo se saiu melhor.
>
> No final é gerado um gráfico com o resultado de erro absoluto médio de cada modelo utilizado para facilitar a comparação.
