public class LogisticRegression{
  public double[] sigmoid(double[] params, double[][] inputs){
    double[] z = Matrices.multiply(params, inputs);
    double[] predictions = new double[inputs[0].length];
    for(int i = 0; i < predictions.length; i++){
      predictions[i] = 1.0/(1.0 + Math.exp(z[i]));
    }
    return predictions;
  }

  public double cost(double[][] inputs, double[] outputs, double[] params){
    double[] predictions = sigmoid(params, inputs);
    return Matrices.sum(Matrices.add(Matrices.multiplyPairwise(outputs, ln(predictions, false)),
     Matrices.multiplyPairwise(Matrices.add(1,Matrices.multiply(-1,outputs)), ln(predictions, true))));
  }

  public double[] ln(double[] predictions, boolean oneMinus){
    for(int i = 0; i < predictions.length; i++){
      if(oneMinus){
        predictions[i] = Math.log(1-predictions[i]);
      } else{
        predictions[i] = Math.log(predictions[i]);
      }
    }
    return predictions;
  }
}
