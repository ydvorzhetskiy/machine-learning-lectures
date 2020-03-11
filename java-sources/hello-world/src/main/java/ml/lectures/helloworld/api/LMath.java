package ml.lectures.helloworld.api;

/**
 * LMath  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface LMath {

    /**
     * Calculate deviation
     * @param actual - actual value
     * @param ideal - ideal value
     * @return deviation
     */
    double deviation(double actual, double ideal);

    /**
     * Logistic function
     * @param x - input
     * @return some value
     */
    double activation(double x);

    /**
     * Delta for weight
     * @param grad - weight gradient
     * @param delta - previous delta
     * @return some value
     */
    double dweight(double grad, double delta);

    /**
     * Gradient for neuron
     * @param out - actual output
     * @return delta for outputs
     */
    double gradient(double out, double delta);

    /**
     * Delta for neuron:
     * @param out - OUT
     * @param weight - Wi
     * @param delta - OUT.delta
     * @return delta for outputs
     */
    double dneuron(double out,
                   double[] weight,
                   double[] delta);

    /**
     * Delta for outputs:
     * @param actual - actual value
     * @param ideal - ideal value
     * @return delta for outputs
     */
    double doutput(double actual, double ideal);
}