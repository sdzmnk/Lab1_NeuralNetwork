using System;
using System.Linq;

namespace Lab1_Zimenko_TV21
{
    internal class Program
    {

        private static double _min;
        private static double _max;

        static void Main()
        {
            NeuralNetwork nn = new NeuralNetwork(_inputSize: 3, _hiddenSize: 2, _outputSize: 1, _learningRate: 0.01);

            double[][] trainInputs =
            {
                new double[] { 0.87, 4.12, 0.93 },
                new double[] { 4.12, 0.93, 4.62 },
                new double[] { 0.93, 4.62, 1.51 },
                new double[] { 4.62, 1.51, 5.76 },
                new double[] { 1.51, 5.76, 0.50 },
                new double[] { 5.76, 0.50, 5.48 },
                new double[] { 0.50, 5.48, 0.95 },
                new double[] { 5.48, 0.95, 4.03 },
                new double[] { 0.95, 4.03, 0.92 },
                new double[] { 4.03, 0.92, 5.15 },
            };

            double[][] trainOutputs =
            {
                new double[] { 4.62 },
                new double[] { 1.51 },
                new double[] { 5.76 },
                new double[] { 0.50 },
                new double[] { 5.48 },
                new double[] { 0.95 },
                new double[] { 4.03 },
                new double[] { 0.92 },
                new double[] { 5.15 },
                new double[] { 1.66 },
            };

            NormalizeData(trainInputs, trainOutputs);


            Console.WriteLine("Training neural network...");
            nn.Train(trainInputs, trainOutputs, epochs: 1000000, errorThreshold: 0.001);

            double[][] testInputs =
            {
                new double[] { 0.92, 5.15, 1.66 },
                new double[] { 5.15, 1.66, 5.01 }
            };

            double[][] expectedOutputs =
            {
                new double[] { 5.01 },
                new double[] { 0.40 }
            };

            NormalizeData(testInputs, expectedOutputs);

            Console.WriteLine("\nTesting neural network...");
            for (int i = 0; i < testInputs.Length; i++)
            {
                double[] result = nn.Predict(testInputs[i]);

                double[] denormalizedInputs = DenormalizeData(testInputs[i]);
                double[] denormalizedResult = DenormalizeData(result);
                double[] denormalizedExpected = DenormalizeData(expectedOutputs[i]);

                nn.PrintResults(denormalizedInputs, denormalizedResult, denormalizedExpected);
            }
        }

        static void NormalizeData(double[][] inputs, double[][] outputs)
        {
            _min = inputs.Concat(outputs).SelectMany(x => x).Min();
            _max = inputs.Concat(outputs).SelectMany(x => x).Max();

            NormalizeArray(inputs);
            NormalizeArray(outputs);
        }

        static void NormalizeArray(double[][] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = (data[i][j] - _min) / (_max - _min);
                }
            }
        }

        public static double[] DenormalizeData(double[] data)
        {
            double[] denormalized = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                denormalized[i] = data[i] * (_max - _min) + _min;
            }
            return denormalized;
        }
    }
}
