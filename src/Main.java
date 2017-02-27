import networks.LayerNetwork;
import networks.Network;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int inputSize = 8;
        int numberHiddenLayer = 1;
        int hiddenSize = 5;
        int outputSize = 8;
        Network network = new LayerNetwork(inputSize, numberHiddenLayer, hiddenSize, outputSize);
        List<List<Integer>> input = Arrays.asList(
                Arrays.asList(new Integer[]{1, 0, 0, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 1, 0, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 1, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 1, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 1, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 1, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 0, 1, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 0, 0, 1})
                );
        List<List<Integer>> output = Arrays.asList(
                Arrays.asList(new Integer[]{1, 0, 0, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 1, 0, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 1, 0, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 1, 0, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 1, 0, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 1, 0, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 0, 1, 0}),
                Arrays.asList(new Integer[]{0, 0, 0, 0, 0, 0, 0, 1})
                );

        for(int i = 0; i < 20000; i++) {
            for(int sample = 0; sample < inputSize; sample++) {
                network.trainStochastic(input.get(sample), output.get(sample));
            }
        }
        for(int sample = 0; sample < inputSize; sample++) {
            System.out.println(network.forwardFeedNetwork(input.get(sample)));
        }
    }
}
