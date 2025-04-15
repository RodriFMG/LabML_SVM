from sklearn.svm import SVC
from ExtraerData import ExtractDataIris
from PreProcesamiento import NormalizeFeatures
from Metricas import Metricas

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = ExtractDataIris(0.1)
    x_train, x_test = NormalizeFeatures(x_train, x_test)

    # Kernel lineal
    svm_lineal = SVC(kernel='linear')
    svm_lineal.fit(x_train, y_train)

    # Kernel polinomial
    svm_poly = SVC(kernel='poly', degree=3)  # degree = grado del polinomio.
    svm_poly.fit(x_train, y_train)

    # Kernel sigmoid
    svm_sigmoid = SVC(kernel='sigmoid')
    svm_sigmoid.fit(x_train, y_train)

    print("Desempeño del Kernel Lineal: ")
    Metricas(x_test, y_test, svm_lineal)

    print("\nDesempeño del Kernel Polynomial: ")
    Metricas(x_test, y_test, svm_poly)

    print("\nDesempeño del Kernel Sigmoidal: ")
    Metricas(x_test, y_test, svm_sigmoid)