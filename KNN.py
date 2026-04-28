'''
Trabalho 1 de Aprendizado de Máquina - Classificador KNN sobre o dataset 'predictive_maintenance'

    Desenvolvido em Python, este pipeline lê o dataset e o pré-processa excluindo atributos
    que representam ruído ou redundância (UDI, Product ID, Type, Target);

    O modelo KNN é criado utilizando três métricas de distância: Euclidiana, Manhattan e Chebyshev.
    A acurácia de cada métrica é avaliada via validação cruzada para valores de K de 1 até 15.
    Para cada métrica, o melhor K é selecionado e utilizado para predizer uma nova amostra de teste.
    Ao fim, as três métricas são comparadas por acurácia para identificar a melhor configuração geral.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('T1-Aprendizado-de-M-quina/predictive_maintenance.csv')

print(dataset.columns)

dataset = dataset[~dataset["Failure Type"].isin(["Random Failures", "No Failure"])]

x = dataset.drop(columns=["UDI", "Product ID", "Type", "Target", "Failure Type"]).values
y = dataset["Failure Type"].values

nomes_atributos = dataset.drop(columns=["UDI", "Product ID", "Type", "Target", "Failure Type"]).columns.tolist()
nomes_classes = sorted(dataset["Failure Type"].unique().tolist())

df = pd.DataFrame(x, columns=nomes_atributos)
df['Failure Type'] = y

print("Head do dataset: ")
print(df.head())

sns.pairplot(df, hue="Failure Type")
plt.show()

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("\nTamanho do conjunto de treino: ", x_treino.shape)
print("Tamanho do conjunto de teste: ", x_teste.shape)

scaler = StandardScaler()
x_treino_padronizado = scaler.fit_transform(x_treino)
x_teste_padronizado = scaler.transform(x_teste)

print("\nExemplo de dado original")
print(x_treino[0])

print("\nExemplo de dado padronizado: ")
print(x_treino_padronizado[0])


metricas = ['euclidean', 'manhattan', 'chebyshev']

melhores = {}  

nova_amostra = np.array([[300.0, 310.0, 1500, 40.0, 150]])
nova_amostra_padronizada = scaler.transform(nova_amostra)

for metrica in metricas:
    print(f"\n{'='*50}")
    print(f"Métrica: {metrica.upper()}")
    print(f"{'='*50}")


    knn = KNeighborsClassifier(n_neighbors=3, metric=metrica)
    knn.fit(x_treino_padronizado, y_treino)
    y_pred = knn.predict(x_teste_padronizado)

    print(f"\nAcurácia (k=3): {accuracy_score(y_teste, y_pred):.4f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_teste, y_pred))
    print("\nRelatório de classificação:")
    print(classification_report(y_teste, y_pred, target_names=nomes_classes))

    
    k_values = list(range(1, 16))
    medias = []

    for k in k_values:
        modelo = KNeighborsClassifier(n_neighbors=k, metric=metrica)
        scores = cross_val_score(modelo, x_treino_padronizado, y_treino, cv=5)
        medias.append(scores.mean())

    print("\nMédias de validação para cada K:")
    for k, media in zip(k_values, medias):
        print(f"  K = {k} -> média = {media:.4f}")

    melhor_k = k_values[np.argmax(medias)]
    print(f"\nMelhor K: {melhor_k}")

    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, medias, marker='o')
    plt.title(f'Validação cruzada: desempenho por valor de K ({metrica})')
    plt.xlabel('Valor de K')
    plt.ylabel('Acurácia média')
    plt.grid(True)
    plt.show()

   
    modelo_final = KNeighborsClassifier(n_neighbors=melhor_k, metric=metrica)
    modelo_final.fit(x_treino_padronizado, y_treino)
    y_pred_final = modelo_final.predict(x_teste_padronizado)

    ac_final = accuracy_score(y_teste, y_pred_final)
    print(f"\nAcurácia do modelo final (K={melhor_k}): {ac_final:.4f}")
    print("\nMatriz de confusão do modelo final:")
    print(confusion_matrix(y_teste, y_pred_final))

   
    classe_prevista = modelo_final.predict(nova_amostra_padronizada)
    print(f"\nClasse prevista para a nova amostra: {classe_prevista[0]}")

    melhores[metrica] = {'k': melhor_k, 'acuracia': ac_final, 'modelo': modelo_final}


print(f"\n{'='*50}")
print("COMPARAÇÃO FINAL ENTRE MÉTRICAS")
print(f"{'='*50}")
for metrica, resultado in melhores.items():
    print(f"{metrica:12} -> Melhor K: {resultado['k']:2d} | Acurácia: {resultado['acuracia']:.4f}")

melhor_metrica = max(melhores, key=lambda m: melhores[m]['acuracia'])
print(f"\nMelhor métrica geral: {melhor_metrica.upper()} com acurácia {melhores[melhor_metrica]['acuracia']:.4f}")