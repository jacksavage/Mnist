using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;

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
            Layers.Aggregate(x, (xx, layer) => layer.Predict(xx));

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

        }

        private double Cost(Pattern[] patterns) =>
            patterns.Select(SquaredErrors).Sum() / (2 * patterns.Length);

        private double SquaredErrors(Pattern pattern) =>
            Predict(pattern.X)
            .Zip(pattern.Y, (a, y) => Math.Sqrt(y - a)).Sum();
    }
}
