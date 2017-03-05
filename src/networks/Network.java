package networks;

import java.util.List;

public interface Network {
    List<Integer> forwardFeedNetwork(List<Integer> input);
    void trainStochastic(List<Integer> input, List<Integer> output, double learningRate);
}