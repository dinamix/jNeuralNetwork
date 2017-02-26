package neurons;

public interface Neuron {
    void feedForward();
    void backPropagation(double learningRate);
    double getOutput();
    double getCorrection();
}