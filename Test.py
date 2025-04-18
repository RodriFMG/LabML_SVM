from Variables import TypeModel
from ExtraerData import ExtractDataIris
from PreProcesamiento import NormalizeFeatures
from Metricas import Metricas
import numpy as np


# Testeo de modelos, ejecutaremos una cierta cantidad de veces el modelo y promediaremos los resultados.

def TestingModel(NumIters=1e+2, model="linear"):
    if model not in ['linear', 'sigmoid', 'poly3', 'poly2', 'poly4']:
        raise ValueError(f"Se esperaba un modelo linear, sigmoid o poly, pero se encontró: {model}")

    F1History = []
    AccHistoty = []
    CMHistory = []

    print(f"Ejecutando el modelo: {model}")

    for i in range(int(NumIters)):
        x_train, y_train, x_test, y_test = ExtractDataIris(0.2)
        x_train, x_test = NormalizeFeatures(x_train, x_test)

        SVModel = TypeModel[model]
        SVModel.fit(x_train, y_train)

        CM, F1, ACC = Metricas(x_test, y_test, model=SVModel)

        F1History.append(F1)
        AccHistoty.append(ACC)
        CMHistory.append(CM)

    print(f"Número de iteraciones: {int(NumIters)}")
    print(f"F1-Score promedio: {np.mean(F1History):.4f}")
    print(f"Accuracy promedio: {np.mean(AccHistoty):.4f}")
    print(f"Matriz de confusión promedio: ")

    CM_Prom = np.mean(CMHistory, axis=0)

    for fila in CM_Prom:
        print(" ".join(f"{x:.4f}" for x in fila))

    return F1History, AccHistoty, CMHistory


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def GraficoComparativo(MetricModel1, MetricModel2, SaveFig, MetricModel4 = None, MetricModel5 = None, MetricModel3= None):
    iteraciones = list(range(1, len(MetricModel1) + 1))

    plt.figure(figsize=(10, 6))

    # Dibujar las líneas
    plt.plot(iteraciones, MetricModel1, color='red', label='Linear')
    plt.plot(iteraciones, MetricModel2, color='blue', label='Sigmoid')


    if MetricModel3:
        plt.plot(iteraciones, MetricModel3, color='green', label='Poly G3')
    if MetricModel4:
        plt.plot(iteraciones, MetricModel4, color='black', label='Poly G2')
    if MetricModel5:
        plt.plot(iteraciones, MetricModel5, color='orange', label='Poly G4')

    # Títulos y leyenda
    plt.title('Comparación de Desempeño de Modelos')
    plt.xlabel('Iteraciones')
    plt.ylabel('Evaluación del Modelo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"./Resultados/{SaveFig}")

    plt.show()
