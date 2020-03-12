package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.ArrayWeights;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.LNet;
import ml.lectures.helloworld.api.SigmoidMath;
import ml.lectures.helloworld.api.TrainSet;
import ml.lectures.helloworld.api.Weights;
import org.apache.commons.lang3.mutable.MutableDouble;

import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import static java.lang.String.format;
import static java.lang.System.currentTimeMillis;
import static java.lang.System.out;
import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.TrainCommon.deviation;
import static ml.lectures.helloworld.api.Utils.dump;
import static ml.lectures.helloworld.api.Utils.dumpLegend;
import static ml.lectures.helloworld.api.Utils.randomizeWeights;

/**
 * TrainQuarters2
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainQuarters2 {

    public static void main(String[] args) {

        val net = new H1Net(new SigmoidMath(0.5, 1.1));
        val weights = new ArrayWeights(2, 4, 1);
        randomizeWeights(weights);
        dumpLegend(weights);
        dump(weights);

        final TrainSet set = consumer -> {
            for (double i = 0.; i <= 1.0; i += 0.2) {
                for (double j = 0.; j <= 1.0; j += 0.2) {
                    consumer.accept(
                        new double[] {i, j}, quarter(i, j)
                    );
                }
            }
        };
        train(net, BPOINTS, set, weights, 10000);

        test(
            net,
            weights,
            arr -> {
                if (arr[3] > 0.1) {
                    out.println(
                        format("[%.3f, %.3f]=[%.3f], E=%.3f", arr[0], arr[1], arr[2], arr[3])
                    );
                }
            }
        );
    }

    private static void test(final LNet net, final Weights weights, Consumer<double[]> consumer) {

        val toConsume = new double[4];
        for (double i = 0.; i <= 1.0; i += 0.1) {
            for (double j = 0.; j <= 1.0; j += 0.1) {
                toConsume[0] = i;
                toConsume[1] = j;
                val r = net.test(weights, new double[] {i, j})[0];
                val e = deviation(r, quarter(i, j)[0]);
                toConsume[2] = r;
                toConsume[3] = e;
                consumer.accept(toConsume);
            }
        }
    }

    static void train(final LNet net,
                      final int[] bpoints,
                      final TrainSet set,
                      final Weights weights,
                      final int epochs) {

        val bp = new HashSet<>();
        for (int i: bpoints) {
            bp.add(i);
        }
        bp.add(epochs);
        val started = currentTimeMillis();
        for (int i = 1; i <= epochs; i++) {
            net.train(weights, set);
            if (bp.contains(i)) {
                val error = new MutableDouble(0.);
                val cnt = new AtomicInteger(0);
                test(net, weights, arr -> {
                    error.add(arr[3]);
                    cnt.incrementAndGet();
                });
                out.println(format("epoch: %d" + "\tE: %.3f", i, error.doubleValue() / cnt.get()));
                dump(weights);
            }
        }
        out.println(format("Timed\t%d", currentTimeMillis() - started));

    }

    //
    //      0  |   1
    //   ------+-------
    //      1  |   0
    //
    private static double[] quarter(final double i, final double j) {

        return new double[] {(i <= 0.5 && j <= 0.5) || (i >= 0.5 && j >= 0.5) ? 1 : 0};
    }
}