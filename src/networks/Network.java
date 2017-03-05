package networks;

import trainers.Trainer;

import java.util.List;

public interface Network {
    List<Double> forwardFeedRounded(List<Double> input);
    List<Double> forwardFeed(List<Double> input);
    void trainStochastic(List<Double> input, List<Double> output, double learningRate);
    void trainStrategy(Trainer trainer, List<List<Double>> inputs, List<List<Double>> outputs,
                       double learningRate, double epsilon);
}