import networks.LayerNetwork;
import networks.Network;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        int inputSize = 8;
        int numberHiddenLayer = 1;
        int hiddenSize = 4;
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

        Random rand = new Random();
        double learningRate = 0.15;
        for(int i = 0; i < 60000; i++) {
            int sample = rand.nextInt(8);
            network.trainStochastic(input.get(sample), output.get(sample), learningRate);
        }
        for(int sample = 0; sample < inputSize; sample++) {
            System.out.println(network.forwardFeedNetwork(input.get(sample)));
        }
    }
}
