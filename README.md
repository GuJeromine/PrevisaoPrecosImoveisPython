# Previsão de Preços de Imóveis com Scikit-learn

Um script de machine learning que utiliza quatro modelos de regressão diferentes (Regressão Linear, Árvore de Decisão, Gradient Boosting e Random Forest) para prever o preço de casas com base em suas características.

## Bibliotecas Utilizadas
* **NumPy:** Para manipulação de arrays numéricos.
* **pandas:** Para carregar e manipular os dados do arquivo CSV.
* **matplotlib:** Para visualizar o gráfico de resultados.
* **scikit-learn:** Para pré-processamento (`SimpleImputer`), divisão dos dados (`train_test_split`), avaliação (`mean_absolute_error`) e os modelos de machine learning.

## Modelos Comparados
* Regressão Linear
* Árvore de Decisão (`DecisionTreeRegressor`)
* Gradient Boosting (`GradientBoostingRegressor`)
* Random Forest (`RandomForestRegressor`)

## O Projeto

### Problema
Muitas vezes, antes de comprar um imóvel, é difícil saber se um preço é justo sem um amplo conhecimento de mercado. Esta solução ajuda a estimar um valor com base em características-chave, facilitando a busca e a tomada de decisão.

### Objetivo/Solução
O objetivo é fazer previsões de preços de casas (`Precio`) lendo um conjunto de dados do arquivo `Casa_toluca.csv`.

Ao contrário do que foi mencionado na descrição anterior, este script **não utiliza a localização**. A previsão é baseada estritamente nas seguintes características:
* `Baños` (Banheiros)
* `Area` (Área)
* `Habitaciones` (Quartos)

## Como Funciona
O script segue um fluxo de trabalho padrão de machine learning:

1.  **Carregamento dos Dados:** O arquivo `Casa_toluca.csv` é carregado em um DataFrame do pandas.
2.  **Pré-processamento:** O `SimpleImputer` é usado para preencher quaisquer valores ausentes (NaN) nas colunas de *features* (X) e na coluna de alvo (y), utilizando a estratégia da média (`mean`).
3.  **Divisão dos Dados:** O conjunto de dados é dividido em 80% para treinamento e 20% para teste (`test_size=0.2`).
4.  **Treinamento:** Os quatro modelos de regressão são treinados com os dados de `x_treino` e `y_treino`.
5.  **Avaliação:** O desempenho de cada modelo é medido calculando o **Erro Absoluto Médio (MAE)** comparando as previsões (`previsoes`) com os valores reais (`y_teste`).
6.  **Previsão de Exemplo:** Além de avaliar, o script também prevê o preço de um novo imóvel com as características `[9 Banheiros, 4000 de Área, 7 Quartos]` para demonstrar o uso prático.
7.  **Visualização:** No final, um gráfico de barras é gerado usando `matplotlib` para comparar visualmente o MAE de cada modelo].

## Como Executar

1.  **Instale as dependências:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```
2.  **Execute o script:**
    Certifique-se de que o arquivo `Casa_toluca.csv` esteja no mesmo diretório que o script `Previsãoprecocasa.py`.
    ```bash
    python Previsãoprecocasa.py
    ```

### Resultado Esperado
Ao executar, o terminal exibirá:
* O Erro Absoluto Médio (MAE) para cada um dos quatro modelos.
* O preço previsto (em U$) para os "novos dados" de exemplo, segundo cada modelo.

Uma janela pop-up também será aberta, mostrando o gráfico de barras que compara o MAE dos modelos.
