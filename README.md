#Logistic Regression

##Description
This class performs logistic regression and has functions useful for various aspects of classification

##Methods

Method | Return Type | Description
-------|-------------|------------
cost(double[][] inputs, double[] outputs, double[] params, double regularizationConst) | double | Returns the cost of the evaluated predictions compared to the real output values.
gradientAndStep(double[][] dataSetInputs, double[] dataSetOutputs, double[] predictionParameters, double learningRate, double regularizationConst) | double[] | Calculates the gradient with respect to each of the predictionParameters using the dataSetInputs and outputs and updates the parameter with a step of size learningRate in the gradient direction. In order to smooth the prediction, the predictionParameters are also regularized by a factor of regularizationConst.
hypothesis(double[] predictions) | double[] | For each prediction, hypothesis returns the corresponding, rounded binary output 0 or 1 as an array.
ln(double[] predictions, boolean oneMinus) | double[] | Returns an array of the natural logarithm of each prediction if oneMinus is set to false. Otherwise it returns the natural logarithm of 1-prediction.
optimizeParams(double[][] inputs, double[] outputs, double[] params, double learningRate, double regularizationConst, double threshold) | double[] | Optimizes prediction parameters till convergence. Returns optimal prediction parameters.
sigmoid(double[][] dataSet, double[] params) | double[] | Evaluates the sigmoid function of each data point in the dataSet array using a vector of hypothesis parameters. The dataSet should be formatted with each datapoint represented as a column of the input variables.

## Setup/Installation Requirements ##
* Java 7
* Gradle

## Setup/Installation ##
Depending on your project, you will likely only need the LogisticRegression.java file located in the src/main/java/ path. The gradle project environment is only to do unit testing. So, if you do not care for the tests, here's what you need to do:

####Setup without Tests
* Clone this repository to your computer
* Within the cloned repository, navigate to the src/main/java/ file.
* Take the LogisticRegression.java file from this location and put it into the same directory as your java file that needs this code.
* Refer to Static Methods section of this README to know which methods to use for your needs.


If you're starting a new project or want to tinker with the code, I recommend you use the project setup that I have provided. The tests will alert you if something has gone wrong with the Matrices class and you can easily set up new tests for whatever endeavor you are embarking on. Testing is good.

#### Setup with Tests
* Clone this repository to your computer
* Create new java classes for your project in the src/main/java/ directory
* Create new test classes for your project in the src/test/java/ directory
* Use src/main/java/App.java for user interface


## Support and Contact Details ##

Please report any bugs or issues you find to grantsrb@gmail.com

### Legal

Copyright (c) 2016 Satchel Grant

Licensed under the MIT license
