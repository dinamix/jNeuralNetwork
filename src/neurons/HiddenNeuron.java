package neurons;

import networks.Dir;
import networks.DirEdge;
import networks.EdgeMatrix;
import predictors.Predictor;

public class HiddenNeuron implements Neuron {

    private double output; //This gets set during a feed forward through sigmoid function
    private double correction;
    private Predictor predict;
    private EdgeMatrix edgeMatrix;

    public HiddenNeuron(Predictor predict, EdgeMatrix edgeMatrix) {
        this.predict = predict;
        this.edgeMatrix = edgeMatrix;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
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
        for(Neuron out : edgeMatrix.getNeuronConnections(this)) {
            DirEdge dirEdgeOut = edgeMatrix.getDirEdge(this, out);
            if(dirEdgeOut.getDir().equals(Dir.IN)) continue; //if coming into this then continue
            double outWeight = dirEdgeOut.getWeight();
            for (Neuron in : edgeMatrix.getNeuronConnections(this)) {
                DirEdge dirEdgeIn = edgeMatrix.getDirEdge(this, in);
                if(dirEdgeIn.getDir().equals(Dir.OUT)) continue; //if coming out of this then continue
                double currentWeight = dirEdgeIn.getWeight();
                correction = computeCorrection(outWeight, out.getCorrection());
                double newWeight = correctWeight(currentWeight, output, correction, learningRate);
                edgeMatrix.updateEdge(in, this, newWeight);
            }
        }
    }

    public double correctWeight(double currentWeight, double output, double correction, double learningRate) {
        return currentWeight + learningRate * correction * output;
    }

    public double computeCorrection(double outWeight, double outCorrection) {
        return output * (1.0 - output) * outWeight * outCorrection;
    }
}
