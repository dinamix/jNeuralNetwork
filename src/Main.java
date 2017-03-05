import networks.LayerNetwork;
import networks.Network;
import trainers.GradientDescent;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int inputSize = 8;
        int numberHiddenLayer = 1;
        int hiddenSize = 4;
        int outputSize = 8;
        //TODO should be able to strategy pattern a loss function into network
        Network network = new LayerNetwork(inputSize, numberHiddenLayer, hiddenSize, outputSize);
        List<List<Double>> input = Arrays.asList(
                Arrays.asList(new Double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})
        );
        List<List<Double>> output = Arrays.asList(
                Arrays.asList(new Double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}),
                Arrays.asList(new Double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})
        );
        double learningRate = 0.85;
        double epsilon = 0.0005;
        long time = System.currentTimeMillis();
        network.trainStrategy(new GradientDescent(), input, output, learningRate, epsilon);
        System.out.println("Learned output for identity matrix input: ");
        for(int sample = 0; sample < inputSize; sample++) {
            System.out.println(network.forwardFeedRounded(input.get(sample)));
        }
        System.out.println("Time to run " + (System.currentTimeMillis() - time) / 1000.0 + "s");
    }
}
