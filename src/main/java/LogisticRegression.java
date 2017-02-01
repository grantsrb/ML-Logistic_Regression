
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

  public double[] hypothesis(double[] predictions){
    for(int i = 0; i < predictions.length; i++){
      predictions[i] = predictions[i] >= 0.5 ? 1.0 : 0.0;
    }
    return predictions;
  }

  // True processes log(1-x) whereas false does log(x)
  public double[] ln(double[] predictions, boolean oneMinus){
    double[] newPredictions = new double[predictions.length];
    for(int i = 0; i < predictions.length; i++){
      if(oneMinus){
        newPredictions[i] = Math.log(1-predictions[i]);
      } else{
        newPredictions[i] = Math.log(predictions[i]);
      }
    }
    return newPredictions;
  }

  public double cost(double[][] inputs, double[] outputs, double[] params){
    double[] predictions = sigmoid(inputs, params);
    double[] outs1 = Matrices.multiplyPairwise(outputs, ln(predictions, false));
    double[] outs0 = Matrices.multiplyPairwise(Matrices.add(1,Matrices.multiply(-1,outputs)), ln(predictions, true));
    double[] outs = Matrices.add(outs1,outs0);
    return (-1.0/outputs.length)*Matrices.sum(outs);
  }

  public double[] gradientAndStep(double[][] inputs, double[] outputs, double[] params, double learningRate){
    double[] predictions = sigmoid(inputs, params);
    double[] updatedParams = new double[params.length];
    for(int i = 0; i < params.length; i++) {
      for(int j = 0; j < outputs.length; j++) {
        double sum = Matrices.sum(Matrices.multiplyPairwise(Matrices.subtract(predictions, outputs), inputs[i]));
        updatedParams[i] = params[i] - learningRate/outputs.length*sum;
      }
    }
    return updatedParams;
  }

  public double[] optimizeParams(double[][] inputs, double[] outputs, double[] params, double learningRate){
    return null;
  }

}
