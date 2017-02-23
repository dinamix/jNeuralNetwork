package neurons;

public interface Neuron {
    void feedForward();
    void backPropagation(double learningRate);
    double correctWeight(double currentWeight, double x, double correction, double learningRate);
    double computeCorrection();
}