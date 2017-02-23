package neurons;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Ugo on 22/02/2017.
 */
public class InputNeuron implements Neuron {

    private double x;
    private Map<HiddenNeuron, Double> neighborWeights;

    public InputNeuron() {
        neighborWeights = new HashMap<>();
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setNeighbor(HiddenNeuron hidden, double weight) {
        neighborWeights.put(hidden, weight);
    }

    @Override
    public void feedForward() {

    }

    @Override
    public void backPropagation(double learningRate) {

    }

    @Override
    public double correctWeight(double currentWeight, double x, double correction, double learningRate) {
        return 0;
    }

    @Override
    public double computeCorrection() {
        return 0;
    }

}
