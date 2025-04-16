from Test import TestingModel, GraficoComparativo

if __name__ == "__main__":
    F1Linear, AccLinear, CMLinear = TestingModel(model='linear')
    print("\n")
    F1Sigmoid, AccSigmoid, CMSigmoid = TestingModel(model='sigmoid')
    print("\n")
    F1Poly, AccPoly, CMPoly = TestingModel(model='poly')

    GraficoComparativo(F1Linear, F1Sigmoid, F1Poly, "F1Score")
    GraficoComparativo(AccLinear, AccSigmoid, AccPoly, "Accuracy")
