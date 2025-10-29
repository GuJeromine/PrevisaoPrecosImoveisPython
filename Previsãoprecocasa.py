import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Carrega os dados
dados = pd.read_csv("Casa_toluca.csv")  

# Processamento dos dados
columns_y = ["Precio"]
features_y = dados[list(columns_y)].values
imp_y = SimpleImputer(missing_values=np.nan, strategy='mean') # Remover valores nan
y = imp_y.fit_transform(features_y)

columns_x = ["Baños", "Area", "Habitaciones"]
features_x = dados[list(columns_x)].values
imp_x = SimpleImputer(missing_values=np.nan, strategy='mean') # Remover valores nan
x = imp_x.fit_transform(features_x)

# Divide os dados em conjuntos de treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Altera a dimensão do vetor
y_treino = y_treino.reshape(-1)
y_teste = y_teste.reshape(-1)

# Inicializa o modelo de regressão linear
modelo = LinearRegression()
# Inicializa o modelo de árvore de decisão
modelo_arvore = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2)
# Inicializa o modelo de Random Forest
modelo_rf = RandomForestRegressor(n_estimators=10, random_state=42)
# Inicializa o modelo de Gradient Boosting
modelo_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Treina os modelos
modelo.fit(x_treino, y_treino)
modelo_arvore.fit(x_treino, y_treino)
modelo_rf.fit(x_treino, y_treino)
modelo_gb.fit(x_treino, y_treino)

# Faz previsões, para o cálculo do erro absoluto
previsoes = modelo.predict(x_teste)
previsoes_arvore = modelo_arvore.predict(x_teste)
previsoes_rf = modelo_rf.predict(x_teste)
previsoes_gb = modelo_gb.predict(x_teste)

# Faz previsões para novos dados
novos_dados = np.array([9, 4000, 7]).reshape(1, -1) # Dados indicados devem ser inseridos aqui, reshape(1, -1) para remodelar a array

# Avalia o desempenho de cada modelo pelo cálculo do erro absoluto médio e mostra os preços previstos em cada modelo
mae = mean_absolute_error(y_teste, previsoes)
print("Erro absoluto médio (Regressão Linear):", mae)
preco_previsto = modelo.predict(novos_dados)
print("Preço previsto da casa (Regressão Linear): U$", preco_previsto[0], "\n")

mae_arvore = mean_absolute_error(y_teste, previsoes_arvore)
print("Erro absoluto médio (Árvore de Decisão):", mae_arvore)
preco_previsto_arvore = modelo_arvore.predict(novos_dados)
print("Preço previsto da casa (Árvore de Decisão): U$", preco_previsto_arvore[0], "\n")

mae_rf = mean_absolute_error(y_teste, previsoes_rf)
print("Erro absoluto médio (Random Forest):", mae_rf)
preco_previsto_rf = modelo_rf.predict(novos_dados)
print("Preço previsto da casa (Random Forest): U$", preco_previsto_rf[0], "\n")

mae_gb = mean_absolute_error(y_teste, previsoes_gb)
print("Erro absoluto médio (Gradient Boosting):", mae_gb)
preco_previsto_gb = modelo_gb.predict(novos_dados)
print("Preço previsto da casa (Gradient Boosting): U$", preco_previsto_gb[0])

# Plota o gráfico com todos os erros absolutos médios
modelos = ["Regressão Linear", "Árvore de Decisão", "Gradient Boosting", "Random Forest"]
resultados = [mae, mae_arvore, mae_gb, mae_rf]

plt.figure(figsize=(8, 6)) # Ajusta a janela de forma que tudo esteja visível
plt.bar(modelos, resultados, color=["purple", "red", "orange", "yellow"])
plt.title("Erro Absoluto Médio")
plt.xlabel("Modelos")
plt.ylabel("Resultados MAE")
plt.show()