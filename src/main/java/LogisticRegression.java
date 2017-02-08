
public class LogisticRegression{

  public LogisticRegression() {}

  public double[] sigmoid(double[][] inputs, double[] params){
    double[] z = Matrices.multiply(params, inputs);
    double[] predictions = new double[z.length];
    for(int i = 0; i < predictions.length; i++){
      predictions[i] = 1.0/(1.0 + Math.exp(-z[i]));
    }
    return predictions;
  }

  public double[][] sigmoid(double[][] inputs, double[][] params){
    double[][] predictions = new double[params.length][inputs[0].length];
    for(int j = 0; j < params.length; j++) {
      double[] z = Matrices.multiply(params[j], inputs);
      for(int i = 0; i < predictions[0].length; i++){
        predictions[j][i] = 1.0/(1.0 + Math.exp(-z[i]));
      }
    }
    return predictions;
  }

  public double[] hypothesis(double[] predictions){
    for(int i = 0; i < predictions.length; i++){
      predictions[i] = predictions[i] >= 0.5 ? 1.0 : 0.0;
    }
    return predictions;
  }

  // True evaluates as log(1-x) whereas false is log(x)
  public double[] ln(double[] predictions, boolean oneMinus){
    double[] newPredictions = new double[predictions.length];
    for(int i = 0; i < predictions.length; i++){
      if(oneMinus){
        if(1-predictions[i] <= 0){
          newPredictions[i] = -1000;
        } else {
          newPredictions[i] = Math.log(1-predictions[i]);
        }

      } else{
        if(predictions[i] == 0){
          newPredictions[i] = -1000;
        } else {
          newPredictions[i] = Math.log(predictions[i]);
        }
      }
    }
    return newPredictions;
  }

  public double cost(double[][] inputs, double[] outputs, double[] params, double regularizationConst){
    double[] predictions = sigmoid(inputs, params);
    double[] logCosts1 = Matrices.multiplyPairwise(outputs, ln(predictions, false));
    double[] logCosts2 = Matrices.multiplyPairwise(Matrices.add(1,Matrices.multiply(-1,outputs)), ln(predictions, true));
    double[] logCosts = Matrices.add(logCosts1,logCosts2);
    double[] regularizationVec = Matrices.multiplyPairwise(params, params);
    regularizationVec[0] = 0;
    return (-1.0/outputs.length)*Matrices.sum(logCosts) +
            regularizationConst/2.0/outputs.length*Matrices.sum(regularizationVec);
  }

  public double[] gradientAndStep(double[][] inputs, double[] outputs, double[] params, double learningRate, double regularizationConst){
    double[] predictions = sigmoid(inputs, params);
    double[] updatedParams = new double[params.length];
    for(int i = 0; i < params.length; i++) {
      for(int j = 0; j < outputs.length; j++) {
        double sum = Matrices.sum(Matrices.multiplyPairwise(Matrices.subtract(predictions, outputs), inputs[i]));
        updatedParams[i] =
            params[i]*(1.0-learningRate*regularizationConst/outputs.length)
            - learningRate/outputs.length*sum;
      }
    }
    return updatedParams;
  }

  public double[] optimizeParams(double[][] inputs, double[] outputs, double[] params, double learningRate, double regularizationConst, double threshold){
    double kost = cost(inputs, outputs, params, regularizationConst);
    double lastKost = kost + 100;
    while(Math.abs(kost-lastKost) > threshold){
      lastKost = kost;
      params = gradientAndStep(inputs, outputs, params, learningRate, regularizationConst);
      kost = cost(inputs, outputs, params, regularizationConst);
    }
    return params;
  }

}
