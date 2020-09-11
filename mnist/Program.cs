using mnist;
using Serilog;
using System;
using System.Linq;

// send logs to console
Log.Logger = new LoggerConfiguration().MinimumLevel.Verbose().WriteTo.Console().CreateLogger();
Log.Information("Let's train a neural net to predict digits from the MNIST handwritten digit images");

var baseUrl = @"http://yann.lecun.com/exdb/mnist/";

// load the training data
Log.Debug("Loading training data");
var train =
    Pattern.LoadPatterns(
        baseUrl, 
        imageFile: @"train-images-idx3-ubyte.gz",
        labelFile: @"train-labels-idx1-ubyte.gz"
    );

// load the testing data
Log.Debug("Loading testing data");
var test =
    Pattern.LoadPatterns(
        baseUrl, 
        imageFile: @"t10k-images-idx3-ubyte.gz",
        labelFile: @"t10k-labels-idx1-ubyte.gz"
    );

// create a shared random number generator
var rand = new Random();

// create a network
var layerSizes = new int[] { 784, 6, 10 };
Log.Debug("Generating a neural net with layer sizes of {LayerSizes}", layerSizes);
var network = new Network(layerSizes);

// train the network
var finalCost =
    network.Train(epochs: 100, batchSize: 1000, learnRate: 0.1, train, test)
    .TakeWhile(cost => cost > 0.001)
    .Last();
Log.Information("Done training");

Log.CloseAndFlush();
System.Diagnostics.Debugger.Break();
