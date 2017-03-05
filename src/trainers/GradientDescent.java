package trainers;

import networks.Network;

import java.util.List;
import java.util.Random;

/**
 * Created by Ugo on 05/03/2017.
 */
public class GradientDescent implements Trainer {
    @Override
    public void train(Network network, List<List<Double>> inputs, List<List<Double>> outputs, double learningRate, double epsilon) {
        Random rand = new Random();
        double error = Double.MAX_VALUE;
        while(error > epsilon) {
            int sample = rand.nextInt(inputs.size());
            List<Double> thisInput = inputs.get(sample);
            List<Double> thisOutput = outputs.get(sample);
            network.trainStochastic(thisInput, thisOutput, learningRate);
            error = computeError(network, thisInput, thisOutput);
        }
    }

    @Override
    public double computeError(Network network, List<Double> input, List<Double> output) {
        List<Double> realOutput = network.forwardFeed(input);
        double error = 0.0;
        for(int i = 0; i < output.size(); i++) {
            error += Math.pow((output.get(i)) - realOutput.get(i), 2) / 2.0;
        }
        return error;
    }
}
