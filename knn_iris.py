"""
Implementacao didatica do algoritmo KNN (K-Nearest Neighbors)
Dataset: Iris (Fisher, 1936)
Trabalho Academico de Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    nomes_classes = iris.target_names

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    k = 5
    modelo = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_pred)
    matriz = confusion_matrix(y_teste, y_pred)

    print("=" * 55)
    print(f"  RESULTADOS DO MODELO KNN (K = {k})")
    print("=" * 55)
    print(f"\nAcuracia no conjunto de teste: {acuracia:.4f} ({acuracia*100:.2f}%)\n")
    print("Matriz de Confusao:")
    print(matriz)
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_teste, y_pred, target_names=nomes_classes))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=nomes_classes
    )
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusao - KNN (K={k})")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png", dpi=120)
    print("Grafico salvo em 'matriz_confusao.png'.")


if __name__ == "__main__":
    main()
