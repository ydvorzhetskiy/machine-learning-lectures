package ml.lectures.helloworld;

/**
 * LMachine  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface LMachine {

    /**
     * train
     * @param set - training set
     * @param ideal - ideal data
     */
    void train(double[][] set, double[] ideal);

    /**
     * weights count
     * @return weights count
     */
    int wsize();

    /**
     * vertex count
     * @return vertex count
     */
    int vsize();
}