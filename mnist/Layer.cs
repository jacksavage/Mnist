using System;
using static mnist.Utilities;

namespace mnist
{
    public record Layer
    {
        public Neuron[] Neurons { get; }
        public Func<double, double> Activate { get; }

        public Layer(int sizeIn, int sizeOut)
        {
            Neurons = ArrayFill(sizeOut, () => new Neuron(sizeIn));

            // todo paramaterize activation function?
            // maybe an enum lookup so so the network is serializable
            Activate = Sigmoid;
        }

        public (double[] a, double[] z) Predict(double[] x)
        {
            var a = new double[Neurons.Length];
            var z = new double[Neurons.Length];

            
            for (var i = 0; i < Neurons.Length; i++)
            {
                z[i] = Neurons[i].Predict(x);
                a[i] = Activate(z[i]);
            }

            return (a, z);
        }
    }
}
