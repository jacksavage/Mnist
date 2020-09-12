using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using static mnist.Utilities;

namespace mnist
{
    public class Network
    {
        public Layer[] Layers { get; }

        public Network(int[] sizes)
        {
            Layers = new Layer[sizes.Length - 1];

            for (var i = 0; i < sizes.Length - 2; i++)
                Layers[i] = new Layer(sizeIn: sizes[i], sizeOut: sizes[i + 1]);
        }

        public double[] Predict(double[] x) =>
            Layers.Aggregate(x, (xx, layer) => layer.Predict(xx).a);

        public IEnumerable<double> Train(int epochs, int batchSize, double learnRate, Pattern[] train, Pattern[] test)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var cost = TrainEpoch(batchSize, learnRate, train, test);
                Log.Information("Epoch {Epoch}, Cost {Cost}", epoch, cost);
                yield return cost;
            }
        }


        private double TrainEpoch(int batchSize, double learnRate, Pattern[] train, Pattern[] test)
        {
            // shuffle the data
            Log.Verbose("Shuffling training data");
            train.Shuffle();

            // update the network weights and biases for each batch
            Log.Verbose("Training on each batch");
            for (var i = 0; i < train.Length; i += batchSize)
                TrainBatch(
                    learnRate, 
                    batchSize, 
                    patterns: train.SelectRange(i, batchSize)
                );

            // evaluate testing data for average cost
            return Cost(test);
        }

        private void TrainBatch(double learnRate, int batchSize, IEnumerable<Pattern> patterns)
        {
            // accumulate cost gradient for every pattern in batch
            var gradW = Layers.Select(L => ArrayFill(L.Neurons.Length, () => new double[L.SizeIn])).ToArray();
            var gradB = Layers.Select(L => new double[L.Neurons.Length]).ToArray();
            foreach (var pattern in patterns)
                Backprop(pattern, gradW, gradB);

            // todo update the weights and biases in the network by a factor of batch size and learning rate
        }

        private void Backprop(Pattern pattern, double[][][] gradW, double[][] gradB)
        {
            // forward pass, storing pre & post activation
            var aL = pattern.X;
            var a = new double[Layers.Length][];
            var z = new double[Layers.Length][];
            for (var i = 0; i < Layers.Length; i++)
            {
                var prediction = Layers[i].Predict(aL);
                z[i] = prediction.z;
                a[i] = prediction.a;
            }

            // find output error
            // 'z' not needed for sigmoid prime bc it can be written as a function of 'a'
            var L = Layers.Length - 1;
            var costPrime = ArrayZip(a[L], pattern.Y, (aa, yy) => aa - yy);
            var sigmoidPrime = ArraySelect(a[L], aa => aa * (1 - aa));
            var delta = ArrayZip(costPrime, sigmoidPrime, (c, s) => c * s);

            // for each neuron, accumulate this error in the gradient for the final layer
            for (var neuronI = 0; neuronI < Layers[L].Neurons.Length; neuronI++)
            {
                // gradB = delta
                gradB[L][neuronI] += delta[neuronI];
                
                // gradW = ain * dout
                for (var weightI = 0; weightI < Layers[L].SizeIn; weightI++)
                    gradW[L][neuronI][weightI] += delta[neuronI] * a[L - 1][weightI];
            }

            // todo find delta for every previous layer, then accumulate that error in the gradient as well
        }

        private double Cost(Pattern[] patterns) =>
            patterns.Select(SquaredErrors).Sum() / (2 * patterns.Length);

        private double SquaredErrors(Pattern pattern) =>
            Predict(pattern.X)
            .Zip(pattern.Y, (a, y) => Math.Sqrt(y - a))
            .Sum();
    }
}
