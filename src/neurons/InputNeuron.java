package neurons;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class InputNeuron implements Neuron {

    private double output;

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    @Override
    public double getCorrection() {
        throw new NotImplementedException();
    }

    @Override
    public void feedForward() {
        throw new NotImplementedException();
    }

    @Override
    public void backPropagation(double learningRate) {
        throw new NotImplementedException();
    }

}
