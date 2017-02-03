# Classifier
A basic artificial neural network implementation, using the well-known multilayer perceptron model.

# Usage
```cs
// Example gesture config that I quickly made for my own project
var config = ClassifierFactory.Config.Gesture;

// 30 patterns (I am using mostly positive examples, some negative)
var trainData = new double[30][];

// Classification of each pattern {0, 1}
var targets = new int[30];

// A forward thrust might look something like this...
// Note: example gesture config uses 11 (x, y, z) positions
trainData[0] = new double[]
{
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.1,
  0.0, 0.0, 0.2,
  0.0, 0.0, 0.3,
  ...
  0.0, 0.0, 0.10
};

// This is a positive example
targets[0] = 1;

// ...

var callback = delegate(Classifier classifier)
{
  classifier.Classify(trainData[0]); // Use a new sample pattern here instead
};

ClassifierFactory.CreateGestureClassifier(config, trainData, targets, classifier => ...);
```
In reality, you of course do not want to supply training data in such a way. Using my project as an example, I record relative positions of the user's hand over a short time, using a positive:negative pattern ratio of around 2:1.

# Remarks
I use this within a small part of a university project, so really it is a WIP. Configuring the network is based on early MLP research and trial and error. Take a look at ```ClassifierFactory``` for setting up your own configuration.
