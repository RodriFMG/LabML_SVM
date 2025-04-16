from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, accuracy_score


def Metricas(x_test, y_test, model):
    y_pred = model.predict(x_test)
    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred, average='macro'), accuracy_score(y_test, y_pred)

