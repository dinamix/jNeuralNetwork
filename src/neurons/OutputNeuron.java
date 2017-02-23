package neurons;

import predictors.Predictor;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Ugo on 22/02/2017.
 */
public class OutputNeuron implements Neuron {

    private double x;
    private Map<HiddenNeuron, Double> neighborWeights;
    private Predictor predict;

    public OutputNeuron(Predictor predict) {
        this.predict = predict;
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
        double sum = 0;
        for(HiddenNeuron input : neighborWeights.keySet()) {
            double weight = neighborWeights.get(input);
            sum += weight * input.getX();
        }
        this.x = predict.predict(sum);
    }

    @Override
    public void backPropagation() {

    }
}
