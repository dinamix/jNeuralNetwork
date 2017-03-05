package neurons;

/*
 * Could make generic if different parameters were required.
 * Would also need to make Network generic as well for training.
 * @param <T>
 */
public interface Neuron<T> {
    void feedForward();
    void backPropagation(double learningRate);
    double getOutput();
    double getCorrection();
}