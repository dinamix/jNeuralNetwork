package neurons;

import networks.Dir;
import networks.DirEdge;
import networks.EdgeMatrix;
import predictors.Predictor;

public class OutputNeuron implements Neuron {

    private double output;
    private double y;
    private double correction;
    private Predictor predict;
    private EdgeMatrix edgeMatrix;

    public OutputNeuron(Predictor predict, EdgeMatrix edgeMatrix) {
        this.predict = predict;
        this.edgeMatrix = edgeMatrix;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getCorrection() {
        return correction;
    }

    @Override
    public void feedForward() {
        double sum = 0;
        for(Neuron input : edgeMatrix.getNeuronConnections(this)) {
            DirEdge dirEdge = edgeMatrix.getDirEdge(this, input);
            if(dirEdge.getDir().equals(Dir.OUT)) continue; //continue if edge coming out of this
            double weight = dirEdge.getWeight();
            sum += weight * input.getOutput();
        }
        this.output = predict.predict(sum);
    }

    @Override
    public void backPropagation(double learningRate) {
        for(Neuron hidden : edgeMatrix.getNeuronConnections(this)) {
            double currentWeight = edgeMatrix.getEdgeWeight(this, hidden);
            correction = computeCorrection();
            double newWeight = correctWeight(currentWeight, hidden.getOutput(), correction, learningRate);
            edgeMatrix.updateEdge(this, hidden, newWeight);
        }
    }

    public double correctWeight(double currentWeight, double x, double correction, double learningRate) {
        return currentWeight + correction * x * learningRate;
    }

    public double computeCorrection() {
        return output * (1.0 - output) * (y - output);
    }
}
