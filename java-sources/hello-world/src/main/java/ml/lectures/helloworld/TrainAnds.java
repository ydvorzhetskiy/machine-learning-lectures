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
import ml.lectures.helloworld.api.ListTrainSet;
import ml.lectures.helloworld.api.SigmoidMath;
import org.apache.commons.lang3.mutable.MutableDouble;

import java.util.stream.Stream;

import static java.util.stream.Collectors.toSet;

/**
 * TrainAnds  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainAnds {

    public static void main(String[] args) {

//        val machine = new H1Net(new SigmoidMath(1.8, 0.8));
        val machine = new H1Net(new SigmoidMath(0.3, 0.7));
        val weights = new ArrayWeights(2, 2, 1)
            .i2h(0, 0, 0.5)
            .i2h(0, 1, 0.3)
            .i2h(1, 0, -0.5)
            .i2h(1, 1, 0.5)
            .h2o(0, 0, 0.2)
            .h2o(1, 0, 0.3)
            .b2h(0, 0, -0.2)
            .b2h(0, 1, 0.2);

        val set = new ListTrainSet()
            .add(new double[] {0, 0}, new double[] {and(0, 0)})
            .add(new double[] {0, 1}, new double[] {and(0, 1)})
            .add(new double[] {1, 1}, new double[] {and(1, 1)})
            .add(new double[] {1, 0}, new double[] {and(1, 0)});

        val bp = Stream.of(1, 20, 30, 1_100, 1_370, 5_000, 10_000)
            .collect(toSet());

        weights.dumpLegend();
        for (int i = 1; i <= 40000; i++) {
            machine.train(weights, set);
            if (bp.contains(i)) {
                System.out.println(String.format("%d", i));
                weights.dump();
                val error = new MutableDouble(0.);
                machine.check(weights, set, error::add);
                System.out.println(String.format("\t%.3f", error.doubleValue()));
//                System.out.println(String.format("\t%.3d", error.longValue()));
//                if (error.doubleValue() < 0.001) {
//                    break;
//                }
            }
        }
    }

    private static int and(final int i, final int j) {
        return i & j;
    }
}