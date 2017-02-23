package networks;

import neurons.InputNeuron;

import java.util.List;

public interface Network {
    void forwardFeedNetwork(List<InputNeuron> input);
}