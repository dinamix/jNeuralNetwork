package networks;

import java.util.List;

public interface Network {
    void forwardFeedNetwork(List<Double> input, List<Double> output);
}