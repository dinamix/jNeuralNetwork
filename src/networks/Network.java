package networks;

import java.util.List;

public interface Network {
    void forwardFeedNetwork(List<Integer> input);
    void trainStochastic(List<Integer> input, List<Integer> output);
}