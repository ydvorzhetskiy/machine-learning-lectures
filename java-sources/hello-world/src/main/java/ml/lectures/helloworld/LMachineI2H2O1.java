package ml.lectures.helloworld;

import lombok.val;

import static java.lang.String.format;
import static java.lang.System.arraycopy;
import static java.lang.System.out;

/**
 * LmachineI2H2O1  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class LMachineI2H2O1 {

    static final int VERTEX_CNT = 7;
    static double epsilon = 0.1;
    static double alpha = 1.1;

    static final int w1 = 0;
    static final int w2 = 1;
    static final int w3 = 2;
    static final int w4 = 3;
    static final int w5 = 4;
    static final int w6 = 5;
    static final int w7 = 6;
    static final int w8 = 7;

    static final int i1 = 0;
    static final int i2 = 1;
    static final int h1 = 2;
    static final int h2 = 3;
    static final int o1 = 4;
    static final int b1 = 5;


    public double[] teach(final double[][] set,
                          final double[] initialWeights,
                          final int epochs) {

        double[] weights = new double[initialWeights.length];
        arraycopy(initialWeights, 0, weights, 0, initialWeights.length);
        for (int i = 0; i < epochs; i++) {
            teachEpoch(weights, set);
        }
        return weights;
    }

    private void teachEpoch(final double[] weights, final double[][] set) {

        val deltas = new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (final double[] ints : set) {
            val outputs = new double[VERTEX_CNT];
            val ideal = ints[2];
            passForward(ints, weights, outputs);
            passBackward(weights, deltas, outputs, deltao(outputs[o1], ideal));
        }
        for (int i = w1; i <= w8; i++) {
            weights[i] = weights[i] + deltas[i];
        }
    }

    private static void passBackward(final double[] weights,
                                     final double[] deltas,
                                     final double[] outputs,
                                     final double deltao) {

        deltas[w5] = deltaw(grad(outputs[h1], deltao), deltas[w5]);
        deltas[w6] = deltaw(grad(outputs[h2], deltao), deltas[w6]);

        val deltaH1 = deltah(outputs[h1], weights[w5], deltao);
        val deltaH2 = deltah(outputs[h2], weights[w6], deltao);

        //h1
        deltas[w1] = deltaw(grad(outputs[i1], deltaH1), deltas[w1]);
        deltas[w3] = deltaw(grad(outputs[i2], deltaH1), deltas[w3]);
        deltas[w8] = deltaw(grad(outputs[b1], deltaH1), deltas[w8]);

        //h2
        deltas[w2] = deltaw(grad(outputs[i1], deltaH2), deltas[w2]);
        deltas[w4] = deltaw(grad(outputs[i2], deltaH2), deltas[w4]);
        deltas[w7] = deltaw(grad(outputs[b1], deltaH2), deltas[w7]);
    }

    public void dumpWeights(final double[] weights) {

        out.println(format("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
            weights[w1], weights[w2], weights[w3],
            weights[w4], weights[w5], weights[w6], weights[w7], weights[w8])
        );
    }

    public  void passForward(final double[] set,
                                    final double[] weights,
                                    final double[] outputs) {

        outputs[i1] = set[0];
        outputs[i2] = set[1];
        outputs[b1] = 1;

        val inp = new double[5];
        inp[h1] = outputs[i1] * weights[w1] + outputs[i1] * weights[w3] + outputs[b1] * weights[w8];
        inp[h2] = outputs[i1] * weights[w2] + outputs[i2] * weights[w4] + outputs[b1] * weights[w7];

        //H-outputs
        outputs[h1] = sigmoid(inp[h1]);
        outputs[h2] = sigmoid(inp[h2]);

        inp[o1] = outputs[h1] * weights[w5] + outputs[h2] * weights[w6];
        outputs[o1] = sigmoid(inp[o1]);
    }

    /**
     * DELTA.w = EPSILON * GRAD.w + ALPHA * DELTA.w-1
     * @param grad - GRAD.w
     * @param delta - DELTA.w-1
     * @return deltaw
     */
    private static double deltaw(final double grad, final double delta) {

        return epsilon * grad + alpha * delta;
    }

    /**
     * Delta for neuron:
     * GRAD.a.b = DELTA.b * OUT.a
     * @param out - actual output
     * @return delta for outputs
     */
    private static double grad(final double out, final double delta) {

        return out * delta;
    }

    /**
     * Delta for neuron:
     * H.delta = F'(IN) * SUM(Wi * OUT.delta)
     * F'(IN) = F.sigmoid = (1 - OUT) * OUT
     * @param out - OUT
     * @param weight - Wi
     * @param delta - OUT.delta
     * @return delta for outputs
     */
    private static double deltah(final double out,
                                 final double weight,
                                 final double delta) {

        return (1 - out) * out * (weight * delta);
    }

    /**
     * Delta for outputs:
     * OUT.delta = (OUT.ideal - OUT.actual) * F'(IN)
     * F'(IN) = F.sigmoid = (1 - OUT.actual) * OUT.actual
     * @param actual - OUT.actual
     * @param ideal - OUT.ideal
     * @return delta for outputs
     */
    static double deltao(final double actual, final double ideal) {

        return (ideal - actual) * (1 - actual) * actual;
    }

    static double error(final double actual, final double ideal) {

        return Math.pow(ideal - actual, 2) / 2.;
    }

    static double sigmoid(final double x) {

        return 1 / (1 + Math.pow(Math.E, (-1 * x)));
    }
}