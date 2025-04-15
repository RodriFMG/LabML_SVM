from sklearn.preprocessing import StandardScaler


# Usaremos el Standar, con este método de normalización se consigue: mean = 0, std = 1
def NormalizarData(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def NormalizeFeatures(x_train, x_test):
    return NormalizarData(x_train), NormalizarData(x_test)
