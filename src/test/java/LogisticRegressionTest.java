import org.junit.*;
import static org.junit.Assert.*;

public class LogisticRegressionTest {
  LogisticRegression lrObj = new LogisticRegression();

  // double[] outs = {1,0,0,1,0,0,0,1,0};


  @Test
  public void sigmoid_runsParamterizationThruSigmoidFxn_doubleArray(){
    double[][] input = {{1,1,1,1,1,1,1,1,1},{1,0,0,1,0,0,0,1,0}};
    double[] params = {0,1};
    double x = 1.0/(1.0+Math.exp(-1));
    double[] expectedOutput = {x,0.5,0.5,x,0.5,0.5,0.5,x,0.5};
    double[] testOutput = lrObj.sigmoid(input,params);
    assertTrue(Matrices.equals(expectedOutput, testOutput));
  }

  @Test
  public void hypothesis_roundsToZeroOrOne_doubleArray(){
    double[] input = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
    double[] expectedOutput = {0,0,0,0,1,1,1,1,1};
    double[] testOutput = lrObj.hypothesis(input);
    assertTrue(Matrices.equals(testOutput, expectedOutput));
  }

  @Test
  public void ln_naturalLogIsAppliedToEachElementWithinAnArray_doubleArray(){
    double[] input = {0.1,0.5,1.0,2.0};
    double w = Math.log(0.1);
    double x = Math.log(0.5);
    double y = Math.log(1.0);
    double z = Math.log(2.0);
    double[] expectedOutput = {w,x,y,z};
    double[] testOutput = lrObj.ln(input, false);
    assertTrue(Matrices.equals(testOutput, expectedOutput));
  }

  @Test
  public void ln_naturalLogIsAppliedTo1minusEachElementWithinAnArray_doubleArray(){
    double[] input = {0.1,0.5,1.0,2.0};
    double w = Math.log(1.0-0.1);
    double x = Math.log(1.0-0.5);
    double y = Math.log(1.0-1.0);
    double z = Math.log(1.0-2.0);
    double[] expectedOutput = {w,x,y,z};
    double[] testOutput = lrObj.ln(input, true);
    assertTrue(Matrices.equals(testOutput, expectedOutput));
  }

  @Test
  public void cost_costFunctionGivenASetOfData_double(){
    double[][] input1 = {{1},{10}};
    double[][] input2 = {{1},{-10}};
    double[] output1 = {1};
    double[] output2 = {0};
    double[] params = {0,1};
    double cost1 = lrObj.cost(input1, output1, params, 0.1);
    double cost2 = lrObj.cost(input2, output2, params, 0.1);
    assertTrue(Math.abs(cost1-cost2) < 0.0001);
  }

  @Test
  public void gradientAndStep_performsASingleOptimizationStepForAllParams_doubleArray(){
    double[][] input = {{1,1,1,1,1,1,1},{1,0,1,0,1,0,1}};
    double[] output = {1,0,1,0,1,0,1};
    double[] params = {1.0,5};
    double[] testOutput = lrObj.gradientAndStep(input, output, params, 0.01, 0.1);
    // assertTrue(testOutput[0] < params[0] && testOutput[1] > params[1]);
  }

  @Test
  public void optimizeParams_performsFullOptimizationOfParameters_doubleArray() {
    double[][] input = {{1,1,1,1,1,1,1},{1,0,1,0,1,0,1}};
    double[] output = {1,0,1,0,1,0,1};
    double[] params = {1.0,1.0};
    double[] testOutput = lrObj.optimizeParams(input, output, params, 0.1, 0.1, 0.000001);
    // Matrices.printArray(testOutput);
    assertTrue(testOutput[0] < params[0] && testOutput[1] > params[1]);
  }
}
