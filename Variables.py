ColumnasDeseadas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm",
                    "Species"]
NumDataIris = 150
NumCamposDeseados = len(ColumnasDeseadas)
PathDB = "./Iris/Iris.csv"

from sklearn.svm import SVC

TypeModel = {
    'linear': SVC(kernel='linear'),
    'sigmoid': SVC(kernel='sigmoid'),
    'poly3': SVC(kernel='poly', degree=3),
    'poly2': SVC(kernel='poly', degree=2),
    'poly4': SVC(kernel='poly', degree=4)
}