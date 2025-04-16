from Test import TestingModel, GraficoComparativo

if __name__ == "__main__":
    F1Linear, AccLinear, CMLinear = TestingModel(model='linear')
    print("\n")
    F1Sigmoid, AccSigmoid, CMSigmoid = TestingModel(model='sigmoid')
    print("\n")
    F1Poly3, AccPoly3, CMPoly3 = TestingModel(model='poly3')
    print("\n")
    F1Poly2, AccPoly2, CMPoly2 = TestingModel(model='poly2')
    print("\n")
    F1Poly4, AccPoly4, CMPoly4 = TestingModel(model='poly4')

    GraficoComparativo(F1Linear, F1Sigmoid, SaveFig="F1Score",
                       MetricModel3=F1Poly3, MetricModel4=None, MetricModel5=None)
    GraficoComparativo(AccLinear, AccSigmoid, SaveFig="Accuracy",
                       MetricModel3=AccPoly3, MetricModel4=None, MetricModel5=None)
