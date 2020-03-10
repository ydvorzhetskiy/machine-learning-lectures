package ml.lectures.helloworld.api;

import java.util.function.BiConsumer;

/**
 * TrainSet  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface TrainSet {

    /**
     * forEach
     * @param consumer - consumer
     */
    void forEach(BiConsumer<double[], double[]> consumer);

    /**
     * add
     * @param data - set
     * @param target - target
     * @return this
     */
    TrainSet add(double[] data, double[] target);
}