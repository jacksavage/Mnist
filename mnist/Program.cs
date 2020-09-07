using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;

// load our data
var baseUrl = @"http://yann.lecun.com/exdb/mnist/";
var trainData =
    ReadData(
        baseUrl, 
        imageFile: @"train-images-idx3-ubyte.gz",
        labelFile: @"train-labels-idx1-ubyte.gz"
    ).ToArray();
var testData = 
    ReadData(
        baseUrl, 
        imageFile: @"t10k-images-idx3-ubyte.gz",
        labelFile: @"t10k-labels-idx1-ubyte.gz"
    ).ToArray();

// instantiate a random number generator
var rand = new Random((int)(DateTime.Now.Ticks % int.MaxValue));

// create a network
var (x1, y1) = trainData.First();
var layerSizes = new int[] { x1.Length, 6, y1.Length };
var (weights, biases) = InitializeNetwork(rand, layerSizes);

// train the network, reporting performance for each epoch
var epochInfo =
    Train(
        weights,
        biases,
        trainData,
        testData,
        rand,
        numEpochs: 100,
        learnRate: 0.1,
        batchSize: 1000
    );

// print the performance on each epoch
foreach (var (epoch, numCorrect) in epochInfo)
    Console.WriteLine($"epoch {epoch} : {numCorrect} / {testData.Length}");
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

static (double[][][] weights, double[][] biases) InitializeNetwork(Random rand, int[] layerSizes)
{
    double[] vector(int m) => Enumerable.Range(0, m).Select(i => rand.NextDouble()).ToArray();
    double[][] matrix(int m, int n) => Enumerable.Range(0, n).Select(i => vector(m)).ToArray();
    
    // generate the network weights
    var weights =
        Enumerable.Range(0, layerSizes.Length - 1)
        .Select(i => matrix(m: layerSizes[i], n: layerSizes[i + 1]))
        .ToArray();

    // generate the network biases
    var biases = layerSizes.Skip(1).Select(vector).ToArray();

    return (weights, biases);
}

static void Shuffle<T>(Random rand, T[] data)
{
    for (var n = data.Length; n > 1; n--)
    {
        var k = rand.Next(n);
        var d = data[n];
        data[n] = data[k];
        data[k] = d;
    }
}

static IEnumerable<T> ArrayRange<T>(T[] source, int start, int end)
{
    for (int i = start; i <= end; i++) yield return source[i];
}

static IEnumerable<(int epoch, int numCorrect)> Train(
    double[][][] weights, double[][] biases,
    (double[] x, double[] y)[] trainData, (double[] x, double[] y)[] testData,
    Random rand,
    int numEpochs, double learnRate, int batchSize
) {
    // for each epoch
    for (var epoch = 0; epoch < numEpochs; epoch++)
    {
        // shuffle the data
        Shuffle(rand, trainData);

        // update the weights and biases with each batch
        for (var i = 0; i < trainData.Length; i += batchSize)
        {
            // select the next set of patterns
            // use them to make one adjustment to the weights and biases
            var batch = ArrayRange(trainData, i, i + batchSize - 1);
            TrainBatch(weights, biases, batch, learnRate, batchSize);
        }

        // determine the number of correctly predicted patterns
        var numCorrect = Evaluate(weights, biases, testData);
        yield return (epoch, numCorrect);
    }
}

static void TrainBatch(
    double[][][] weights, double[][] biases,
    IEnumerable<(double[] x, double[] y)> data,
    double learnRate, int batchSize
) {

}

static int Evaluate(
    double[][][] weights, double[][] biases,
    IEnumerable<(double[] x, double[] y)> data
) {

}
