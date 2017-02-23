package networks;

import neurons.HiddenNeuron;
import neurons.InputNeuron;
import neurons.OutputNeuron;
import predictors.LogisticPredictor;
import predictors.Predictor;

import java.util.ArrayList;
import java.util.List;

//TODO Assume we have fully connected bipartite graph between layers for now
//TODO Should change to more flexible network
//TODO Also assume we only have one hidden layer for now, List<List> makes it ready for more later
public class LayerNetwork implements Network{
    private List<InputNeuron> inputLayer;
    private List<OutputNeuron> outputLayer;
    private List<List<HiddenNeuron>> hiddenLayers;

    public LayerNetwork(int inputSize, int hiddenLayerSize, int hiddenSize, int outputSize) {
        inputLayer = new ArrayList<>(inputSize);
        for(int i = 0; i < inputSize; i++) {
            inputLayer.add(new InputNeuron());
        }
        outputLayer = new ArrayList<>(outputSize);
        for(int i = 0; i < outputSize; i++) {
            outputLayer.add(new OutputNeuron(new LogisticPredictor()));
        }
        hiddenLayers = new ArrayList<>(hiddenLayerSize);
        for(int i = 0; i < hiddenLayerSize; i++) {
            List<HiddenNeuron> hidden = new ArrayList<>(hiddenSize);
            for(int h = 0; h < hiddenSize; h++) {
                hidden.add(new HiddenNeuron(new LogisticPredictor()));
            }
            hiddenLayers.add(hidden);
        }
    }

    private void connectNetwork() {
        //Connect input to hidden layer
        for(InputNeuron input : inputLayer) {
            List<HiddenNeuron> firstLayer = hiddenLayers.get(0);
            for(HiddenNeuron hidden : firstLayer) {
                double initialWeight = Math.random();
                hidden.setInput(input, initialWeight);
                input.setNeighbor(hidden, initialWeight);
            }
        }
        //Connect hidden layer to output
        for(OutputNeuron output : outputLayer) {
            List<HiddenNeuron> firstLayer = hiddenLayers.get(0);
            for(HiddenNeuron hidden : firstLayer) {
                double initialWeight = Math.random();
                hidden.setOutput(output, initialWeight);
                output.setNeighbor(hidden, initialWeight);
            }
        }
    }

    @Override
    public void forwardFeedNetwork(List<InputNeuron> input) {
        for(int i = 0; i < input.size(); i++) {
            InputNeuron layerInput = inputLayer.get(i);
            layerInput.setX(input.get(i).getX());
        }
        for(List<HiddenNeuron> layer : hiddenLayers) {
            for(HiddenNeuron hidden : layer) {
                hidden.feedForward();
            }
        }
        for(OutputNeuron output : outputLayer) {
            output.feedForward();
        }
    }
}
