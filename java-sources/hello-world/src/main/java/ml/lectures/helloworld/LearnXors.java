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

/**
 * LearnXors  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class LearnXors {

    public static void main(String[] args) {

        val machine = new LMachineI2H2O1(0.9, 0.9);
        val initialWeights = new double[] {0.5, 0.3, -0.5, 0.5, 0.2, 0.3, 0.0, 0.0};
        final double [][] set = {
            {0, 0, 0 ^ 0},
            {0, 1, 0 ^ 1},
            {1, 1, 1 ^ 1},
            {1, 0, 1 ^ 0}
        };
        double[] weights = machine.teach(set, initialWeights, 5000);
        machine.checkResults("OR", weights, set);
        machine.dumpWeights(weights);
    }

}