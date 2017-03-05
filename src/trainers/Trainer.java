package trainers;

import networks.Network;

import java.util.List;

/**
 * Created by Ugo on 05/03/2017.
 */
public interface Trainer {
    void train(Network network, List<List<Double>> inputs, List<List<Double>> outputs, double learningRate, double epsilon);
    double computeError(Network network, List<Double> input, List<Double> output);
}
