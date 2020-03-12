package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.ArrayWeights;
import ml.lectures.helloworld.api.LNet;
import ml.lectures.helloworld.api.TrainSet;
import ml.lectures.helloworld.api.Weights;

import java.util.HashSet;
import java.util.function.Function;

import static java.lang.String.format;
import static java.lang.System.currentTimeMillis;
import static java.lang.System.out;
import static ml.lectures.helloworld.api.Utils.dump;
import static ml.lectures.helloworld.api.Utils.dumpLegend;

/**
 * TrainCommon  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainCommon {

    public static final int[] BPOINTS = {1, 100, 500, 1_000, 2_500, 5_000, 10_000};

    static Weights weights() {
        return new ArrayWeights(2, 2, 1)
            .i2h(0, 0, 0.5)
            .i2h(0, 1, 0.3)
            .i2h(1, 0, -0.5)
            .i2h(1, 1, 0.5)
            .h2o(0, 0, 0.2)
            .h2o(1, 0, 0.3)
            .b2h(0, 0, -0.2)
            .b2h(0, 1, 0.2);
    }

    static Weights weights2() {
        return new ArrayWeights(2, 3, 1)
            .i2h(0, 0, 0.5)
            .i2h(0, 1, 0.3)
            .i2h(1, 0, -0.5)
            .i2h(1, 1, 0.5)
            .h2o(0, 0, 0.2)
            .h2o(1, 0, 0.3)
            .b2h(0, 0, -0.2)
            .b2h(0, 1, 0.2);
    }

    static final double[][] INPUTS = {
        {0, 0},
        {0, 1},
        {1, 1},
        {0, 0}
    };

    static TrainSet trainSet(final Function<double[], Double> fun) {


        return consumer -> {

            for (double[] set: INPUTS) {
                consumer.accept(set, new double[] {fun.apply(set)});
            }
        };
    }

    public static void train(final LNet net,
                             final int[] bpoints,
                             final TrainSet set,
                             final Weights weights,
                             final int epochs,
                             final Function<double[], Double> fun) {

        val bp = new HashSet<>();
        for (int i: bpoints) {
            bp.add(i);
        }
        bp.add(epochs);

        dumpLegend(weights);
        val started = currentTimeMillis();
        for (int i = 1; i <= epochs; i++) {
            net.train(weights, set);
            if (bp.contains(i)) {
                double error = 0.;
                val errors = new double[INPUTS.length];
                for (int j = 0; j < INPUTS.length; j++) {
                    val in = INPUTS[j];
                    val res = net.test(weights, INPUTS[j]);
                    errors[j] = deviation(res[0], fun.apply(in));
                }
                for (double e: errors) {
                    error += e;
                }
                out.println(format("epoch: %d\terror:\t%.3f", i, error));
                for (double e: errors) {
                    out.println(format("%.3f", e));
                }
                dump(weights);
            }
        }
        out.println(format("Timed\t%d", currentTimeMillis() - started));
    }


    public static double deviation(double actual, double ideal) {
        return Math.pow(ideal - actual, 2) / 2.;
    }
}