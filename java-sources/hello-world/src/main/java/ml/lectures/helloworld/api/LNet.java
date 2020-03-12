package ml.lectures.helloworld.api;

/**
 * LMachine  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface LNet {

    /**
     * train
     * @param weights - weights
     * @param set - set
     */
    void train(Weights weights, TrainSet set);

    /**
     * check
     * @param weights - weights
     * @param input - input
     * @return output
     */
    double[] test(Weights weights, double[] input);
}