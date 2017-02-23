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
    public void backPropagation(double learningRate) {
        for(OutputNeuron out : outputNeighborWeights.keySet()) {
            double currentWeight = outputNeighborWeights.get(out);
            double newWeight = correctWeight(currentWeight, x, computeCorrection(), learningRate);
            outputNeighborWeights.put(out, newWeight);
        }
    }

    @Override
    public double correctWeight(double currentWeight, double x, double correction, double learningRate) {
        //Assuming 1 output neuron
        return currentWeight + learningRate * correction * x;
    }

    @Override
    public double computeCorrection() {
        OutputNeuron output = null;
        for(OutputNeuron out : outputNeighborWeights.keySet()) {
            output = out;
        }
        double w = outputNeighborWeights.get(this);
        double o = predict.predict(x * w);
        return o * (1.0 - o) * w * output.computeCorrection();
    }
}
