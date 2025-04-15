from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, accuracy_score


def Metricas(x_test, y_test, model):
    y_pred = model.predict(x_test)
    print(f"Confusión de matriz: {confusion_matrix(y_test, y_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='macro')}")
    # Con string no funciona xd.
    # print(f"Error Cuadratico Medio: {mean_squared_error(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")