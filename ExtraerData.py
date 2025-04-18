import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Variables import ColumnasDeseadas, NumDataIris, PathDB


# porcentaje de datos que tendrá el test.
def ExtractDataIris(porcTest=0.2):

    if porcTest <= 0 or porcTest >= 1:
        raise ValueError("Se debe pasar un porcentaje en el rango <0, 1>")

    #np.random.seed(37)

    pdIris = pd.read_csv(PathDB)

    TotalData = np.empty((NumDataIris, 0))
    for column in ColumnasDeseadas:
        DataColumn = pdIris[column].to_numpy()
        DataColumn = DataColumn.reshape(-1, 1)
        TotalData = np.concatenate([TotalData, DataColumn], axis=1)

    # Permutación aleatoria.
    np.random.shuffle(TotalData)
    x = TotalData[:, :-1]
    y = TotalData[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=porcTest)

    return x_train, y_train, x_test, y_test
