package neurons;

import jdk.internal.util.xml.impl.Input;
import predictors.Predictor;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Ugo on 22/02/2017.
 */
public class HiddenNeuron implements Neuron {

    private double x;
    private Map<InputNeuron, Double> inputNeighborWeights;
    private Map<OutputNeuron, Double> outputNeighborWeights;
    private Predictor predict;

    public HiddenNeuron(Predictor predict) {
        this.predict = predict;
        inputNeighborWeights = new HashMap<>();
        outputNeighborWeights = new HashMap<>();
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setInput(InputNeuron input, double weight) {
        inputNeighborWeights.put(input, weight);
    }

    public void setOutput(OutputNeuron output, double weight) {
        outputNeighborWeights.put(output, weight);
    }

    @Override
    public void feedForward() {
        double sum = 0;
        for(InputNeuron input : inputNeighborWeights.keySet()) {
            double weight = inputNeighborWeights.get(input);
            sum += weight * input.getX();
        }
        this.x = predict.predict(sum);
    }

    @Override
    public void backPropagation() {

    }
}
