package predictors;

public class LogisticPredictor implements Predictor {
    @Override
    public double predict(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
