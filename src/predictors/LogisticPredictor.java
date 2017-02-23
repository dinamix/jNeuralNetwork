package predictors;

/**
 * Created by Ugo on 22/02/2017.
 */
public class LogisticPredictor implements Predictor {
    @Override
    public double predict(double x) {
        return 1 / (1 - Math.exp(-x));
    }
}
