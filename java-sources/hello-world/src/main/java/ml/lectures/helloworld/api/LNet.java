package ml.lectures.helloworld.api;

import java.util.function.Consumer;

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
     * @param set - set
     * @param error - error
     */
    void check(Weights weights, TrainSet set, Consumer<Double> error);
}