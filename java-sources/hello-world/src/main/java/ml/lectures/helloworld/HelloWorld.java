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

    static final int[][] DATASET = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 1, 0},
        {1, 0, 1}
    };

    static final double EPSILON = 0.7;
    static final double ALPHA = 0.3;

    public static void main(String[] args) {

        printDataset();

        double[] weights = {0.5, 0.3, -0.5, 0.5, 0.2, 0.3};
        val hIns = new double[DATASET.length][2];
        val hOuts = new double[DATASET.length][2];

        val oIns = new double[DATASET.length];
        val oOuts = new double[DATASET.length];
        val errs = new double[DATASET.length];
        val oDeltas = new double[DATASET.length];

        for (int i = 0; i < DATASET.length; i++) {

            hIns[i][0] = DATASET[i][0] * weights[0] + DATASET[i][1] * weights[2];
            hIns[i][1] = DATASET[i][0] * weights[1] + DATASET[i][1] * weights[3];

            hOuts[i][0] = sigmoid(hIns[i][0]);
            hOuts[i][1] = sigmoid(hIns[i][1]);

            oIns[i] = hOuts[i][0] * weights[4] + hOuts[i][0] * weights[5];
            oOuts[i] = sigmoid(oIns[i]);
            errs[i] = error(oOuts[i], DATASET[i][2]);
            oDeltas[i] = odelta(oOuts[i], DATASET[i][2]);
        }

        for (int i = 0; i < hIns.length; i++) {
            out.println(format("\nset %d \n----------------", i));
            val shIns = hIns[i];
            val shOuts = hOuts[i];
            out.println("H\tin\t\tout");
            for (int j = 0; j < shIns.length; j++) {
                out.println(format("%d\t%.2f\t%.2f", j + 1, shIns[j], shOuts[j]));
            }

            val soIn = oIns[i];
            val soOut = oOuts[i];

            out.println("\nO\tin\t\tout");
            out.println(format("%d\t%.2f\t%.2f", 1 , soIn, soOut));
            out.println(format("E\t%.2f", errs[i]));
        }

        //calc o-deltas
        for (int i = 0; i < DATASET.length; i++) {
            oDeltas[i] = odelta(oOuts[i], DATASET[i][2]);
        }

        out.println("\nO-deltas");
        for (int i = 0; i < hIns.length; i++) {
            out.println(format("%d\t%.2f", i, oDeltas[i]));
        }

        //calc h-deltas
        val hDeltas = new double[DATASET.length][2];
        for (int i = 0; i < DATASET.length; i++) {
            hDeltas[i][0] = hdelta(hOuts[i][0], new double[] { weights[4] }, new double[] { oDeltas[i] });
            hDeltas[i][1] = hdelta(hOuts[i][1], new double[] { weights[5] }, new double[] { oDeltas[i] });
        }

        out.println("\nH-deltas");
        for (int i = 0; i < hDeltas.length; i++) {
            out.println(format("\nset %d \n----------------", i));
            val hDelta = hDeltas[i];
            out.println("H\tD");
            for (int j = 0; j < hDelta.length; j++) {
                out.println(format("%d\t%.2f", j + 1, hDelta[j]));
            }
        }

        //calc gradients
        val hGrads = new double[DATASET.length][2];
        for (int i = 0; i < DATASET.length; i++) {
            hGrads[i][0] = grad(hOuts[i][0], oDeltas[i]);
            hGrads[i][1] = grad(hOuts[i][1], oDeltas[i]);
        }

        out.println("\nH-gradients");
        for (int i = 0; i < hDeltas.length; i++) {
            out.println(format("\nset %d \n----------------", i));
            val grads = hGrads[i];
            out.println("H\tG");
            for (int j = 0; j < grads.length; j++) {
                out.println(format("%d\t%.2f", j + 1, grads[j]));
            }
        }

        //calc deltas for weights
        val deltaws = new double[DATASET.length][2];
        for (int i = 0; i < DATASET.length; i++) {
            deltaws[i][0] = deltaw(hGrads[i][0], 0);
            deltaws[i][1] = deltaw(hGrads[i][1], 0);
        }

        out.println("\nW-deltas");
        for (int i = 0; i < deltaws.length; i++) {
            out.println(format("\nset %d \n----------------", i));
            val dw = deltaws[i];
            for (int j = 0; j < dw.length; j++) {
                out.println(format("%d\t%.2f", j + 1, dw[j]));
            }
        }
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
    };

    /**
     * Delta for neuron:
     * H.delta = F'(IN) * SUM(Wi * OUT.delta)
     * F'(IN) = F.sigmoid = (1 - OUT) * OUT
     * @param out - actual output
     * @param weights - weights
     * @return delta for outputs
     */
    private static double hdelta(final double out, final double[] weights, final double[] deltas) {

        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * deltas[i];
        }
        return (1 - out) * out * sum;
    }


    /**
     * Delta for outputs:
     * OUT.delta = (OUT.ideal - OUT.actual) * F'(IN)
     * F'(IN) = F.sigmoid = (1 - OUT) * OUT
     * @param actual - actual output
     * @param ideal - ideal output
     * @return delta for outputs
     */
    private static double odelta(final double actual, final double ideal) {

        return (ideal - actual) * (1 - actual) * actual;
    }

    private static double error(final double actual, final double ideal) {

        return Math.pow(ideal - actual, 2);
    }

    private static void printDataset() {

        for (int[] i: DATASET) {
            out.println(format("%d ^ %d = %d", i[0], i[1], i[2]));
        }
    }

    public static double sigmoid(double x) {

        return 1 / (1 + Math.pow(Math.E, (-1 * x)));
    }
}