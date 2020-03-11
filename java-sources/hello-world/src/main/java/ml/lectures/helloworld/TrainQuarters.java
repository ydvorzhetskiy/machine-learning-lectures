/*
 *  Copyright 2020 Russian Post
 *
 * This source code is Russian Post Confidential Proprietary.
 * This software is protected by copyright. All rights and titles are reserved.
 * You shall not use, copy, distribute, modify, decompile, disassemble or reverse engineer the software.
 * Otherwise this violation would be treated by law and would be subject to legal prosecution.
 * Legal use of the software provides receipt of a license from the right holder only.
 */
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

import static java.lang.String.format;
import static java.lang.System.currentTimeMillis;
import static java.lang.System.out;
import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.api.Utils.randomizeWeights;

/**
 * TrainQuarters  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainQuarters {

    public static void main(String[] args) {

        val net = new H1Net(new SigmoidMath(2.7, 0.1));
        val weights = new ArrayWeights(2, 4, 1);
        randomizeWeights(weights);

        final TrainSet set = consumer -> {
            for (double i = 0.; i <= 1.0; i += 0.2) {
                for (double j = 0.; j <= 1.0; j += 0.2) {
                    consumer.accept(
                        new double[] {i, j}, new double[] {quarter(i, j)}
                    );
                }
            }
        };
        train(net, BPOINTS, set, weights, 50000);
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
                net.check(weights, set,
                    e -> {
                        cnt.incrementAndGet();
                        error.add(e);
                    }
                );
                out.println(format("epoch: %d\terror:\t%.3f", i, error.doubleValue() / cnt.get()));
            }
        }
        out.println(format("Timed\t%d", currentTimeMillis() - started));

    }

    //   0   |   1
    // ------+-------
    //   2   |   3
    //
    private static double quarter(final double i, final double j) {

        if (i <= 0.5) {
            if (j <= 0.5) {
                return 2;
            }
            return 0;
        } else {
            if (j <= 0.5) {
                return 3;
            }
            return 1;
        }
    }
}