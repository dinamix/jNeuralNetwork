package neurons;

public interface Neuron<T> {
    void feedForward();
    void backPropagation(double learningRate);
    double getOutput();
    double getCorrection();
}