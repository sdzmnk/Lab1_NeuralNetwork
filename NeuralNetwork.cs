using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Lab1_Zimenko_TV21
{
    class NeuralNetwork
    {
        private int inputSize, hiddenSize, outputSize;
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;
        private double[] hiddenLayer;
        private double[] outputLayer;
        private double learningRate;

        public NeuralNetwork(int _inputSize, int _hiddenSize, int _outputSize, double _learningRate)
        {
            inputSize = _inputSize;
            hiddenSize = _hiddenSize;
            outputSize = _outputSize;
            learningRate = _learningRate;

            weightsInputHidden = new double[inputSize, hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            hiddenLayer = new double[hiddenSize];
            outputLayer = new double[outputSize];

            Random rnd = new Random();
            InitializeWeights(weightsInputHidden, rnd);
            InitializeWeights(weightsHiddenOutput, rnd);
        }

        private void InitializeWeights(double[,] weights, Random rnd)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = rnd.NextDouble() - 0.5;
                }
            }
        }

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        private double SigmoidDerivative(double x) => x * (1 - x);
        public double[] Predict(double[] inputs)
        {
            CalculateLayer(inputs, weightsInputHidden, hiddenLayer);
            CalculateLayer(hiddenLayer, weightsHiddenOutput, outputLayer);
            return outputLayer;
        }

        private void CalculateLayer(double[] inputs, double[,] weights, double[] layer)
        {
            for (int i = 0; i < layer.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    sum += inputs[j] * weights[j, i];
                }
                layer[i] = Sigmoid(sum);
            }
        }

        public void Train(double[][] trainInputs, double[][] trainOutputs, int epochs, double errorThreshold)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;


                for (int i = 0; i < trainInputs.Length; i++)
                {
                    double[] inputs = trainInputs[i];
                    double[] expected = trainOutputs[i];
                    double[] outputs = Predict(inputs);

                    double[] outputErrors = new double[outputSize];
                    double[] hiddenErrors = new double[hiddenSize];

                    for (int j = 0; j < outputSize; j++)
                    {
                        outputErrors[j] = (expected[j] - outputs[j]) * SigmoidDerivative(outputs[j]);
                        totalError += Math.Pow(expected[j] - outputs[j], 2);
                    }

                    for (int j = 0; j < hiddenSize; j++)
                    {
                        double error = 0;
                        for (int k = 0; k < outputSize; k++)
                            error += outputErrors[k] * weightsHiddenOutput[j, k];
                        hiddenErrors[j] = error * SigmoidDerivative(hiddenLayer[j]);
                    }

                    UpdateWeights(weightsHiddenOutput, hiddenLayer, outputErrors);
                    UpdateWeights(weightsInputHidden, inputs, hiddenErrors);

                    if (epoch % 10000 == 0)
                    {
                        double[] denormalizedInputs = Program.DenormalizeData(inputs);
                        double[] denormalizedExpected = Program.DenormalizeData(expected);
                        double[] denormalizedOutputs = Program.DenormalizeData(outputs);

                        Console.WriteLine($"Epoch: {epoch}");
                        PrintResults(inputs, outputs, expected);
                    }
                }

                double mse = totalError / trainInputs.Length;

                if (epoch % 10000 == 0)
                {
                    Console.WriteLine($" MSE: {mse:F6}\n");
                }

                if (mse < errorThreshold)
                {
                    Console.WriteLine($"Training stopped early at epoch {epoch}: MSE threshold reached. MSE: {mse:F6}");
                    return;
                }
            }
        }
        private void UpdateWeights(double[,] weights, double[] inputs, double[] errors)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] += learningRate * errors[j] * inputs[i];
                }
            }
        }

        public void PrintResults(double[] inputs, double[] outputs, double[] expected)
        {
            Console.WriteLine("Input: " + string.Join("; ", inputs.Select(x => x.ToString("F2"))));
            Console.WriteLine("Expected: " + string.Join("; ", expected.Select(x => x.ToString("F2"))));
            Console.WriteLine("Predicted: " + string.Join("; ", outputs.Select(x => x.ToString("F2"))));

            if (expected != null)
            {
                double error = expected[0] - outputs[0];
                Console.WriteLine($"Error: {error:F6}");
            }
            Console.WriteLine("----------------------");
        }
    }
}
