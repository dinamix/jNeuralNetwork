package networks;

import neurons.HiddenNeuron;
import neurons.InputNeuron;
import neurons.Neuron;
import neurons.OutputNeuron;
import predictors.LogisticPredictor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//TODO Assume we have fully connected bipartite graph between layers for now
//TODO Should change to more flexible network
//TODO Also assume we only have one hidden layer for now, List<List> makes it ready for more later
public class LayerNetwork implements Network{
    private List<InputNeuron> inputLayer;
    private List<OutputNeuron> outputLayer;
    private List<List<HiddenNeuron>> hiddenLayers;
    private EdgeMatrix edgeMatrix;

    public LayerNetwork(int inputSize, int hiddenLayerSize, int hiddenSize, int outputSize) {
        edgeMatrix = new EdgeMatrix();

        //Add nodes to network
        inputLayer = new ArrayList<>(inputSize);
        for(int i = 0; i < inputSize; i++) {
            inputLayer.add(new InputNeuron());
        }
        outputLayer = new ArrayList<>(outputSize);
        for(int i = 0; i < outputSize; i++) {
            outputLayer.add(new OutputNeuron(new LogisticPredictor(), edgeMatrix));
        }
        hiddenLayers = new ArrayList<>(hiddenLayerSize);
        for(int i = 0; i < hiddenLayerSize; i++) {
            List<HiddenNeuron> hidden = new ArrayList<>(hiddenSize);
            for(int h = 0; h < hiddenSize; h++) {
                hidden.add(new HiddenNeuron(new LogisticPredictor(), edgeMatrix));
            }
            hiddenLayers.add(hidden);
        }

        //Connect network
        connectNetwork();
    }

    /**
     * Assume fully connected bipartite graph between layers for now.
     * Also assume that we only have 1 hidden layer for now.
     */
    private void connectNetwork() {
        //Keep a random gaussian distribution through weight initialization
        Random random = new Random();

        //Connect input to hidden layer
        for(InputNeuron input : inputLayer) {
            //TODO account for multiple layers here if needed
            List<HiddenNeuron> firstLayer = hiddenLayers.get(0);
            for(HiddenNeuron hidden : firstLayer) {
                double initialWeight = random.nextGaussian();
                edgeMatrix.updateEdge(input, hidden, initialWeight);
            }
        }

        //Connect hidden layer to output
        for(OutputNeuron output : outputLayer) {
            //TODO account for multiple layers here if needed
            List<HiddenNeuron> firstLayer = hiddenLayers.get(0);
            for(HiddenNeuron hidden : firstLayer) {
                double initialWeight = random.nextGaussian();
                edgeMatrix.updateEdge(hidden, output, initialWeight);
            }
        }
    }

    @Override
    public void forwardFeedNetwork(List<Double> input, List<Double> output) {
        //Copy inputs to network inputs
        for(int i = 0; i < input.size(); i++) {
            InputNeuron layerInput = inputLayer.get(i);
            layerInput.setOutput(input.get(i));
        }

        //Feed to hidden layer
        //TODO Note that this considers multiple hidden layers but in rest of code we assume 1 for now
        for(List<HiddenNeuron> layer : hiddenLayers) {
            for(HiddenNeuron hidden : layer) {
                hidden.feedForward();
            }
        }

        //Get final output
        for(OutputNeuron out : outputLayer) {
            out.feedForward();
        }
    }
}
