package ml.lectures.helloworld.api;

/**
 * SigmoidMath  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class SigmoidMath implements LMath {

    private final double epsilon;
    private final double alpha;

    /**
     * Constructor
     * @param epsilon - learning speed
     * @param alpha - inertia moment
     */
    public SigmoidMath(final double epsilon, final double alpha) {

        this.epsilon = epsilon;
        this.alpha = alpha;
    }

    @Override
    public double activation(final double x) {

        return 1 / (1 + Math.pow(Math.E, (-1 * x)));
    }

    /**
     * DELTA.w = EPSILON * GRAD.w + ALPHA * DELTA.w-1
     * @param grad - GRAD.w
     * @param delta - previous delta
     * @return deltaw
     */
    @Override
    public double dweight(final double grad, final double delta) {

        return epsilon * grad + alpha * delta;
    }

    @Override
    public double grad(final double out, final double delta) {

        return out * delta;
    }

    /**
     * Delta for neuron:
     * H.delta = F'(IN) * SUM(Wi * OUT.delta)
     * or F'(IN) * W1 * OUT.delta + F'(IN) * W2 * OUT.delta
     * F'(IN) = F.sigmoid = (1 - OUT) * OUT
     * @param out - OUT
     * @param weight - Wi
     * @param delta - OUT.delta
     * @return delta for outputs
     */
    @Override
    public double hdelta(final double out, final double[] weight, final double[] delta) {

        double sum = 0.;
        for (int i = 0; i < weight.length; i++) {
            sum += weight[i] * delta[i];
        }
        return (1 - out) * out * sum;
    }

    @Override
    public double hdelta(final double out, final double weight, final double delta) {

        return (1 - out) * out * weight * delta;
    }

    /**
     * Delta for outputs:
     * OUT.delta = (OUT.ideal - OUT.actual) * F'(IN)
     * F'(IN) = F.sigmoid = (1 - OUT.actual) * OUT.actual
     * @param actual - OUT.actual
     * @param ideal - OUT.ideal
     * @return delta for outputs
     */
    @Override
    public double odelta(final double actual, final double ideal) {

        return (ideal - actual) * (1 - actual) * actual;
    }

    /**
     * Derivative for neuron (H-vertex):
     * F'(IN)
     * F'(IN) = F.sigmoid' = (1 - OUT) * OUT
     * @param hout - H-vertex out
     * @return Derivative for H-vertex
     */
    public double hder(final double hout) {
        return (1 - hout) * hout;
    }

    /**
     * H.delta = F'(IN) * SUM(Wi * OUT.delta)
     * @param hder - F'(IN)
     * @param dsum - SUM(Wi * OUT.delta)
     * @return
     */
    public double hdelta(final double hder, final double dsum) {

        return hder * dsum;
    }
}