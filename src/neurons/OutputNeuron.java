package neurons;

import predictors.Predictor;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Ugo on 22/02/2017.
 */
public class OutputNeuron implements Neuron {

    private double x;
    private double y;
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

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
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

    //TODO Probably don't need to use this since the step is performed in the HiddenNeuron
    @Override
    public void backPropagation(double learningRate) {
        for(HiddenNeuron hidden : neighborWeights.keySet()) {
            double currentWeight = neighborWeights.get(hidden);
            //Gradient descent step to update weights
            double newWeight = correctWeight(currentWeight, hidden.getX(), computeCorrection(), learningRate);
            //TODO need to update both weights here since we have 2 nodes per edge
            //TODO should change in one place, this might be easier with a graph library
            neighborWeights.put(hidden, newWeight);
            hidden.setOutput(this, newWeight);
        }
    }

    @Override
    public double correctWeight(double currentWeight, double x, double correction, double learningRate) {
        return currentWeight + correction * x * learningRate;
    }

    @Override
    public double computeCorrection() {
        double o = predict.predict(x);
        return o * (1.0 - o) * (y - o);
    }
}
