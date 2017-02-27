package networks;

public class DirEdge {
    private double weight;
    private Dir dir;

    public DirEdge(double weight, Dir dir) {
        this.weight = weight;
        this.dir = dir;
    }

    public double getWeight() {
        return weight;
    }

    public Dir getDir() {
        return dir;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

}
