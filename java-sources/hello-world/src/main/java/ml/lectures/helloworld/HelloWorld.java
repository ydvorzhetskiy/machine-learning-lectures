package ml.lectures.helloworld;

import lombok.val;

import static java.lang.String.format;
import static java.lang.System.out;

/**
 * HelloWord  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class HelloWorld {

    static final int VERTEX_CNT = 6;
    private static class Model {
        private double[] outputs = new double[VERTEX_CNT];
        private double[] inputs = new double[VERTEX_CNT];
//        outs[i1] = set[0];
//        outs[i2] = set[1];
//
//        val ideal = set[2];
//
//        //weights
//        val inp = new double[5];
    };

    static final double EPSILON = 0.3;
    static final double ALPHA = 0.1;

    static final int w1 = 0;
    static final int w2 = 1;
    static final int w3 = 2;
    static final int w4 = 3;
    static final int w5 = 4;
    static final int w6 = 5;

    static final int i1 = 0;
    static final int i2 = 1;
    static final int h1 = 2;
    static final int h2 = 3;
    static final int o1 = 4;

    public static void main(String[] args) {

        double[] deltas = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] weights = {0.5, 0.3, -0.5, 0.5, 0.2, 0.3};
        int[][] dataset = {
            {0, 0, 0},
            {0, 1, 1},
            {1, 1, 0},
            {1, 0, 1}
        };
        dumpWeights(weights);
        //loop for epoch
        for (int e = 0; e < 2; e++) {
            for (int i = 0; i < dataset.length; i++) {
                doIteration(dataset[i], weights, deltas);
                //summarize
//                dumpWeights(weights);
            }
        }
    }

    private static void dumpWeights(final double[] weights) {
        out.println(format("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
            weights[0], weights[1], weights[2],
            weights[3], weights[4], weights[5]));
    }

    private static void doIteration(final int[] set,
                                    final double[] weights,
                                    final double[] prevDeltas) {

        val outs = new double[5];
        outs[i1] = set[0];
        outs[i2] = set[1];

        val ideal = set[2];

        //weights
        val inp = new double[5];
        inp[h1] = outs[i1] * weights[w1] + outs[i2] * weights[w3];
        inp[h2] = outs[i1] * weights[w2] + outs[i2] * weights[w4];

        //H-outputs
        outs[h1] = sigmoid(inp[h1]);
        outs[h2] = sigmoid(inp[h2]);

        inp[o1] = outs[h1] * weights[w5] + outs[h2] * weights[w6];
        outs[o1] = sigmoid(inp[o1]);

        val err = dev(outs[o1], ideal);
        val hdeltas = new double[5];
        hdeltas[o1] = deltao(outs[o1], ideal);


        // move to upper level to summarize error and deltas
        hdeltas[h1] = deltah(outs[h1], weights[w5], hdeltas[o1]);
        hdeltas[h2] = deltah(outs[h2], weights[w6], hdeltas[o1]);

        val grads = new double[6];
        grads[w5] = grad(outs[h1], hdeltas[o1]);
        grads[w6] = grad(outs[h2], hdeltas[o1]);

        val wdeltas = new double[6];
        wdeltas[w5] = deltaw(grads[w5], prevDeltas[w5]);
        weights[w5] = weights[w5] + wdeltas[w5];

        wdeltas[w6] = deltaw(grads[w6], prevDeltas[w6]);
        weights[w6] = weights[w6] + wdeltas[w5];

        grads[w1] = grad(outs[i1], hdeltas[h1]);
        grads[w2] = grad(outs[i1], hdeltas[h2]);
        grads[w3] = grad(outs[i2], hdeltas[h1]);
        grads[w4] = grad(outs[i2], hdeltas[h2]);

        wdeltas[w1] = deltaw(grads[w1], prevDeltas[w1]);
        wdeltas[w2] = deltaw(grads[w2], prevDeltas[w2]);
        wdeltas[w3] = deltaw(grads[w3], prevDeltas[w3]);
        wdeltas[w4] = deltaw(grads[w4], prevDeltas[w4]);

        weights[w1] = weights[w1] + wdeltas[w1];
        weights[w2] = weights[w2] + wdeltas[w2];
        weights[w3] = weights[w3] + wdeltas[w3];
        weights[w4] = weights[w4] + wdeltas[w4];

        out.println(format("IDEAL:\t%d\tOUT:\t%.3f\tERROR\t%.3f", ideal, outs[o1], err));

        prevDeltas[w1] = wdeltas[w1];
        prevDeltas[w2] = wdeltas[w2];
        prevDeltas[w3] = wdeltas[w3];
        prevDeltas[w4] = wdeltas[w4];
        prevDeltas[w5] = wdeltas[w5];
        prevDeltas[w6] = wdeltas[w6];
    }

    /**
     * DELTA.w = EPSILON * GRAD.w + ALPHA * DELTA.w-1
     * @param grad - GRAD.w
     * @param delta - DELTA.w-1
     * @return deltaw
     */
    private static double deltaw(final double grad, final double delta) {

        return EPSILON * grad + ALPHA * delta;
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

    static double dev(final double actual, final double ideal) {

        return Math.pow(ideal - actual, 2);
    }

    static double sigmoid(double x) {

        return 1 / (1 + Math.pow(Math.E, (-1 * x)));
    }
}