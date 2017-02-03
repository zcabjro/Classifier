using System;

[Serializable]
public class Classifier
{
	[Serializable]
	private class BDN
	{
		private double[] weights;

		public BDN(int numInputs)
		{
			this.weights = new double[1 + numInputs];
		}

		public double Weight(int index)
		{
			return weights[index];
		}

		public void InitWeights(Random random)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] = 0.2 * random.NextDouble() - 0.1;
			}
		}

		public void AdjustWeights(double trainRate, double delta, double[] inputs)
		{
			weights[0] += trainRate * delta; // bias weight
			for (int i = 0; i < inputs.Length; i++)
			{
				weights[i + 1] += trainRate * delta * inputs[i];
			}
		}

		public double Activate(double[] inputs)
		{
			return WeightedSum(inputs);
		}

		private double WeightedSum(double[] inputs)
		{
			double sum = weights[0]; // bias weight
			for (int i = 0; i < inputs.Length; i++)
			{
				sum += inputs[i] * weights[i + 1];
			}
			return sum;
		}
	}

	private BDN[] hiddenNeurons;
	private BDN outputNeuron;

	public double Error
	{
		get
		{
			return error;
		}
	}
	private double error;

	public Classifier(int numInputs, int numHiddenNeurons)
	{
		CreateNeurons(numInputs, numHiddenNeurons);
	}

	private void CreateNeurons(int numInputs, int numHiddenNeurons)
	{
		hiddenNeurons = new BDN[numHiddenNeurons];
		for (int i = 0; i < hiddenNeurons.Length; i++)
		{
			hiddenNeurons[i] = new BDN(numInputs);
		}
		outputNeuron = new BDN(numHiddenNeurons);
	}

	private void InitNeurons(Random random)
	{
		for (int i = 0; i < hiddenNeurons.Length; i++)
		{
			hiddenNeurons[i].InitWeights(random);
		}
		outputNeuron.InitWeights(random);
	}

	public double Train(double[][] trainData, int[] targets, double trainRate, double targetError, int maxEpochs, int maxRestarts)
	{
		// Random used for initialising input weights
		Random random = new Random();

		// Initialise neurons for first training attempt			
		InitNeurons(random);

		int epoch = 0;
		int restarts = 0;
		error = 0.0;
		do
		{
			// For each training pattern, run a forward and backward pass
			for (int i = 0; i < trainData.Length; i++)
			{
				double[] pattern = trainData[i];
				double patternTarget = targets[i];

				// Forward pass
				double[] hiddenNeuronActivations = new double[hiddenNeurons.Length];
				double[] hiddenNeuronOutputs = new double[hiddenNeurons.Length];
				for (int j = 0; j < hiddenNeurons.Length; j++)
				{
					BDN hiddenNeuron = hiddenNeurons[j];
					double activation = hiddenNeuron.Activate(pattern);
					hiddenNeuronActivations[j] = activation;
					hiddenNeuronOutputs[j] = Sigmoid(activation);
				}
				double outputNeuronActivation = outputNeuron.Activate(hiddenNeuronOutputs);
				double outputNeuronOutput = Sigmoid(outputNeuronActivation);

				// Add squared pattern error to total error
				error += Math.Pow(patternTarget - outputNeuronOutput, 2);

				// Backward pass
				double outputDelta = (patternTarget - outputNeuronOutput) * SigmoidDeriv(outputNeuronActivation);
				outputNeuron.AdjustWeights(trainRate, outputDelta, hiddenNeuronOutputs);
				for (int j = 0; j < hiddenNeurons.Length; j++)
				{
					BDN hiddenNeuron = hiddenNeurons[j];
					double hiddenDelta = outputDelta * outputNeuron.Weight(j + 1) * SigmoidDeriv(hiddenNeuronActivations[j]);
					hiddenNeuron.AdjustWeights(trainRate, hiddenDelta, pattern);
				}
			}

			// Square root the average error for a conservative estimate
			error = Math.Sqrt(error / trainData.Length);

			if (++epoch >= maxEpochs)
			{
				// If we run out of attempts, finish now
				if (restarts >= maxRestarts)
				{
					return error;
				}

				// Otherwise, restart the network for another attempt
				InitNeurons(random);
				epoch = 0;
				restarts += 1;
			}

		} while (error > targetError);

		return error;
	}

	public double Classify(double[] pattern)
	{
		double[] hiddenNeuronActivations = new double[hiddenNeurons.Length];
		double[] hiddenNeuronOutputs = new double[hiddenNeurons.Length];
		for (int j = 0; j < hiddenNeurons.Length; j++)
		{
			BDN hiddenNeuron = hiddenNeurons[j];
			double activation = hiddenNeuron.Activate(pattern);
			hiddenNeuronActivations[j] = activation;
			hiddenNeuronOutputs[j] = Sigmoid(activation);
		}
		double outputNeuronActivation = outputNeuron.Activate(hiddenNeuronOutputs);
		double outputNeuronOutput = Sigmoid(outputNeuronActivation);
		return outputNeuronOutput;
	}

	private static double Sigmoid(double x)
	{
		return 1 / (1 + Math.Exp(-x));
	}

	private static double SigmoidDeriv(double x)
	{
		double fx = Sigmoid(x);
		return fx * (1 - fx);
	}
}