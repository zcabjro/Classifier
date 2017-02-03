using System;
using System.Threading;

public static class ClassifierFactory
{
	public enum Config
	{
		// An example configuration for detecting gestures from 3D coordinate patterns
		Gesture
	}

	private class TrainingArgs
	{
		public readonly Classifier Classifier;
		public readonly double[][] TrainData;
		public readonly int[] Targets;
		public readonly double TrainRate;
		public readonly double TargetError;
		public readonly int MaxEpochs;
		public readonly int MaxRestarts;
		public readonly Action<Classifier> Callback;

		public TrainingArgs(
			Classifier classifier,
			double[][] trainData,
			int[] targets,
			double trainRate,
			double targetError,
			int maxEpochs,
			int maxRestarts,
			Action<Classifier> callback)
		{
			Classifier = classifier;
			TrainData = trainData;
			Targets = targets;
			TrainRate = trainRate;
			TargetError = targetError;
			MaxEpochs = maxEpochs;
			MaxRestarts = maxRestarts;
			Callback = callback;
		}
	}

	public static void CreateGestureClassifier(Config config, double[][] trainData, int[] targets, Action<Classifier> callback)
	{
		TrainingArgs trainingArgs;
		switch (config)
		{
			case Config.Gesture:
				trainingArgs = new TrainingArgs(
					classifier: new Classifier(numInputs: 33, numHiddenNeurons: 11),
					trainData: trainData,
					targets: targets,
					trainRate: 0.5,
					targetError: 0.05,
					maxEpochs: 3000,
					maxRestarts: 20,
					callback: callback
				);
				break;
			default:
				throw new ArgumentException("Unhandled config: " + config);
		}

		new Thread(Create).Start(trainingArgs);
	}

	private static void Create(object obj)
	{
		if (!(obj is TrainingArgs))
		{
			throw new ArgumentException("Object is not of type " + typeof(TrainingArgs).FullName);
		}

		TrainingArgs args = (TrainingArgs)obj;
		Classifier classifier = args.Classifier;
		classifier.Train(args.TrainData, args.Targets, args.TrainRate, args.TargetError, args.MaxEpochs, args.MaxRestarts);
		args.Callback(classifier);
	}
}