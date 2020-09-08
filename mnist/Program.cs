using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;

// load our data
var baseUrl = @"http://yann.lecun.com/exdb/mnist/";
var trainingPatterns =
    ReadData(
        baseUrl, 
        imageFile: @"train-images-idx3-ubyte.gz",
        labelFile: @"train-labels-idx1-ubyte.gz"
    ).ToArray();
var testingPatterns = 
    ReadData(
        baseUrl, 
        imageFile: @"t10k-images-idx3-ubyte.gz",
        labelFile: @"t10k-labels-idx1-ubyte.gz"
    ).ToArray();

// instantiate a random number generator
var rand = new Random((int)(DateTime.Now.Ticks % int.MaxValue));

// create a network
var (x0, y0) = trainingPatterns.First();
var layerSizes = new int[] { x0.Length, 6, y0.Length };
var network = InitializeNetwork(rand, layerSizes);

// train the network, reporting performance for each epoch
var epochInfo =
    Train(
        network,
        trainingPatterns,
        testingPatterns,
        rand,
        numEpochs: 100,
        learnRate: 0.1,
        batchSize: 1000
    );

// print the performance on each epoch
foreach (var (epoch, numCorrect) in epochInfo)
    Console.WriteLine($"epoch {epoch} : {numCorrect} / {testingPatterns.Length}");
Console.WriteLine("done training");

System.Diagnostics.Debugger.Break();

static IEnumerable<(double[] x, double[] y)> ReadData(string baseUrl, string imageFile, string labelFile)
{
    return Enumerable.Zip(
        DownloadParseData(baseUrl + imageFile, ReadImages),
        DownloadParseData(baseUrl + labelFile, ReadLabels),
        (x, y) => (x, y)
    ).ToArray();
}

static IEnumerable<double[]> DownloadParseData(string url, Func<BinaryReader, IEnumerable<double[]>> parser)
{
    // download the file if it doesn't exist
    var name = Path.GetFileName(url);
    if (!File.Exists(name))
    {
        using (var client = new WebClient()) client.DownloadFile(url, name);
        if (!File.Exists(name)) yield break;
    }

    // open the file as a binary reader
    using (var file = File.OpenRead(name))
    using (var gzip = new GZipStream(file, CompressionMode.Decompress))
    using (var reader = new BinaryReader(gzip))
        foreach (var vec in parser(reader)) yield return vec;
}

static IEnumerable<double[]> ReadImages(BinaryReader reader)
{
    // read the header
    Func<int> readInt = () => BitConverter.ToInt32(reader.ReadBytes(4).Reverse().ToArray());
    if (readInt() != 2051) yield break;     // magic number
    var n = readInt();                      // sample count
    var h = readInt();                      // image height
    var w = readInt();                      // image width
    var m = h * w;                          // pixels per image

    // read each image
    for (int i = 0; i < n; i++)
    {
        yield return reader
            .ReadBytes(m)                               // read the bytes for each pixel
            .Select(p => p / (double)byte.MaxValue)     // normalize between 0 and 1
            .ToArray();
    }
}

static IEnumerable<double[]> ReadLabels(BinaryReader reader)
{
    // read the header
    Func<int> readInt = () => BitConverter.ToInt32(reader.ReadBytes(4).Reverse().ToArray());
    if (readInt() != 2049) yield break;     // magic number
    var n = readInt();                      // sample count

    // read each label
    for (int i = 0; i < n; i++)
    {
        // convert digit label to a one-hot vector
        var b = reader.ReadByte();
        yield return Enumerable.Range(0, 10)
            .Select(i => b == i ? 1.0 : 0.0)
            .ToArray();
    }
}

static (double[][] weights, double[] biases)[] InitializeNetwork(Random rand, int[] layerSizes)
{
    // generate the network weights and biases
    var network =
        Enumerable.Range(0, layerSizes.Length - 1)
        .Select(i => new { inSize = layerSizes[i], outSize = layerSizes[i + 1] })
        .Select(sizes => (
            weights: Matrix(M: sizes.inSize, N: sizes.outSize, (m, n) => rand.NextDouble()),
            biases: Vector(M: sizes.outSize, m => rand.NextDouble()) 
        )).ToArray();

    return network;
}

static T[] Vector<T>(int M, Func<int, T> elementFunc) => 
    Enumerable.Range(0, M).Select(elementFunc).ToArray();

static T[][] Matrix<T>(int M, int N, Func<int, int, T> elementFunc) =>
    Enumerable.Range(0, N).Select(n => Vector(M, m => elementFunc(m, n))).ToArray();

static double Dot(double[] a, double[] b) => 
    Enumerable.Zip(a, b, (aa, bb) => aa * bb).Sum();

static void Shuffle<T>(T[] patterns, Random rand)
{
    for (var n = patterns.Length; n > 1; n--)
    {
        var k = rand.Next(n);
        var d = patterns[n];
        patterns[n] = patterns[k];
        patterns[k] = d;
    }
}

static IEnumerable<T> ArrayRange<T>(T[] source, int start, int end)
{
    for (int i = start; i <= end; i++) yield return source[i];
}

static IEnumerable<(int epoch, int numCorrect)> Train(
    (double[][] weights, double[] biases)[] network,
    (double[] x, double[] y)[] trainingPatterns, (double[] x, double[] y)[] testingPatterns,
    Random rand,
    int numEpochs, double learnRate, int batchSize
) {
    // for each epoch
    for (var epoch = 0; epoch < numEpochs; epoch++)
    {
        // shuffle the data
        Shuffle(trainingPatterns, rand);

        // update the weights and biases with each batch
        for (var i = 0; i < trainingPatterns.Length; i += batchSize)
        {
            // select the next set of patterns
            // use them to make one adjustment to the weights and biases
            var batch = ArrayRange(trainingPatterns, i, i + batchSize - 1);
            TrainBatch(network, batch, learnRate, batchSize);
        }

        // determine the number of correctly predicted patterns
        var numCorrect = Evaluate(network, testingPatterns);
        yield return (epoch, numCorrect);
    }
}

static void TrainBatch(
    (double[][] weights, double[] biases)[] network,
    IEnumerable<(double[] x, double[] y)> patterns,
    double learnRate, int batchSize
) {

}

static int Evaluate(
    (double[][] weights, double[] biases)[] network,
    IEnumerable<(double[] x, double[] y)> patterns
) {

    int indexOfMax(double[] items) => Enumerable.Range(1, items.Length - 1).Aggregate(0, (iMax, i) => items[i] > items[iMax] ? i : iMax);
    bool isCorrect(double[] x, double[] y) => indexOfMax(Predict(network, x)) == indexOfMax(y);
    return patterns.Count(pattern => isCorrect(pattern.x, pattern.y));
}

static double[] Predict((double[][] weights, double[] biases)[] network, double[] x)
{
    // compute the output of one layer
    // reorganize the weights and biases by neuron
    // for each neuron comput its output
    double[] evalLayer(double[] input, (double[][] weights, double[] biases) layer) =>
        Enumerable.Zip(layer.weights, layer.biases, (weights, bias) => (weights, bias))
        .Select(neuron => Dot(input, neuron.weights) + neuron.bias)
        .ToArray();


    // step through each layer and compute its output
    // final layer output is the network output
    return network.Aggregate(seed: x, evalLayer);
}
