package networks;

import neurons.Neuron;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by Ugo on 25/02/2017.
 */
public class EdgeMatrix {

    //Each Neuron and the edges to corresponding neighbors neurons kept in this state
    private Map<Neuron, Map<Neuron, DirEdge>> edgeMatrix;

    public EdgeMatrix() {
        edgeMatrix = new HashMap<>();
    }

    public Set<Neuron> getNeuronConnections(Neuron n) {
        return edgeMatrix.get(n).keySet();
    }

    /**
     * Assume that edge goes from node n1 to node n2.
     * @param n1
     * @param n2
     * @param weight
     */
    public void updateEdge(Neuron n1, Neuron n2, double weight) {
        if(!edgeMatrix.containsKey(n1)) {
            addNeuron(n1);
        }
        edgeMatrix.get(n1).put(n2, new DirEdge(weight, Dir.OUT));

        if(!edgeMatrix.containsKey(n2)) {
            addNeuron(n2);
        }
        edgeMatrix.get(n2).put(n1, new DirEdge(weight, Dir.IN));
    }

    public double getEdgeWeight(Neuron n1, Neuron n2) {
        if(!edgeMatrix.containsKey(n1) || !edgeMatrix.containsKey(n2)) {
            throw new RuntimeException("Neurons do not exist.");
        }
        return edgeMatrix.get(n1).get(n2).getWeight();
    }

    public DirEdge getDirEdge(Neuron n1, Neuron n2) {
        if(!edgeMatrix.containsKey(n1) || !edgeMatrix.containsKey(n2)) {
            throw new RuntimeException("Neurons do not exist.");
        }
        return edgeMatrix.get(n1).get(n2);
    }

    private void addNeuron(Neuron n) {
        if(!edgeMatrix.containsKey(n)) {
            edgeMatrix.put(n, new HashMap<>());
        }
    }
}
