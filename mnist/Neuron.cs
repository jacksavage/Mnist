using System.Linq;
using static mnist.Utilities;

namespace mnist
{
    public class Neuron
    {
        public double[] Weights { get; }
        public double Bias { get; } 

        public Neuron(int sizeIn)
        {
            Weights = ArrayFill(sizeIn, Rand.NextDouble);
            Bias = Rand.NextDouble();
        }

        public double Predict(double[] x) =>
            Dot(Weights, x) + Bias;

        private static double Dot(double[] a, double[] b) =>
            Enumerable.Zip(a, b, (aa, bb) => aa * bb).Sum();
    }
}
