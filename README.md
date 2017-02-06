Method | Return Type | Description
-------|-------------|--------
sigmoid(double[][] dataSet, double[] params) | double[] | Evaluates the sigmoid function of each data point in the dataSet array using a vector of hypothesis parameters. The dataSet should be formatted with each datapoint represented as a column of the input variables.
hypothesis(double[] predictions) | double[] | For each prediction, hypothesis returns the corresponding, rounded binary output 0 or 1 as an array.
ln(double[] predictions, boolean oneMinus) | double[] | Returns an array of the natural logarithm of each prediction if oneMinus is set to false. Otherwise it returns the natural logarithm of 1-prediction.
cost(double[][] inputs, double[] outputs, double[] params, double regularizationConst) | double | Returns the cost of the evaluated predictions compared to the real output values.
gradientAndStep(double[][] dataSetInputs, double[] dataSetOutputs, double[] predictionParameters, double learningRate, double regularizationConst) | double[] | Calculates the gradient with respect to each of the predictionParameters using the dataSetInputs and outputs and updates the parameter with a step of size learningRate in the gradient direction. In order to smooth the prediction, the predictionParameters are also regularized by a factor of regularizationConst.
optimizeParams(double[][] inputs, double[] outputs, double[] params, double learningRate, double regularizationConst, double threshold) | double[] | Optimizes prediction parameters till convergence. Returns optimal prediction parameters. 
