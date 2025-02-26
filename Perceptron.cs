using System;

class Perceptron
{
    private double[] weights;
    private double learningRate = 0.1;

    public Perceptron(int inputSize, double weight)
    {
        weights = new double[inputSize];
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++)
        {
            //weights[i] = rand.NextDouble() * 2 - 1;
            weights[i] = weight;
        }
    }

    private int ActivationFunction(double x, double threesold) => x >= threesold ? 1 : 0;

    public int Predict(int[] inputs, double threesold)
    {
        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * weights[i];
        }
        return ActivationFunction(sum, threesold);
    }

    //public void Train(int[][] trainingInputs, int[] labels, int epochs, double threesold)
    //{
    //    for (int e = 0; e < epochs; e++)
    //    {
    //        for (int i = 0; i < trainingInputs.Length; i++)
    //        {
    //            int prediction = Predict(trainingInputs[i], threesold);
    //            int error = labels[i] - prediction;
    //            for (int j = 0; j < weights.Length; j++)
    //            {
    //                weights[j] += learningRate * error * trainingInputs[i][j];
    //            }
    //        }
    //    }
    //}
}


