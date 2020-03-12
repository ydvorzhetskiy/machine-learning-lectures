package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.ArrayWeights;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.LNet;
import ml.lectures.helloworld.api.SigmoidMath;
import ml.lectures.helloworld.api.TrainSet;
import ml.lectures.helloworld.api.Weights;

import java.util.HashSet;

import static java.lang.String.format;
import static java.lang.System.currentTimeMillis;
import static java.lang.System.out;
import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.api.Utils.randomizeWeights;

/**
 * TrainHalfs  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainHalfs {

    public static void main(String[] args) {

//        val net = new H1Net(new SigmoidMath(0.5, 1.1));
        val net = new H1Net(new SigmoidMath(0.5, 1.0));
        val weights = new ArrayWeights(1, 2, 1);
        randomizeWeights(weights);

        final TrainSet set = consumer -> {
            for (double i = 0.; i <= 1.0; i += 0.2) {
                consumer.accept(
                    new double[] {i}, half(i)
                );
            }
        };
        train(net, BPOINTS, set, weights, 5000);
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
                val r0 = net.test(weights, new double[] {0.25});
                val r1 = net.test(weights, new double[] {0.75});
                out.println(format("epoch: %d\t[0.25]=[%.3f]\t[0.75]=[%.3f]", i, r0[0], r1[0]));
            }
        }
        out.println(format("Timed\t%d", currentTimeMillis() - started));

    }

    //
    //   0  |  1
    //
    private static double[] half(final double i) {

        return new double[] {i <= 0.5 ? 0 : 1};
    }
}