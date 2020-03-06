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

/**
 * LMath  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface LearnMath {

    /**
     * Calculate deviation
     * @param actual - actual value
     * @param ideal - ideal value
     * @return deviation
     */
    double deviation(double actual, double ideal);

    /**
     * Logistic function
     * @param x - input
     * @return some value
     */
    double logisticFun(double x);

    /**
     * Delta for weight
     * @param grad - weight gradient
     * @param delta - previous delta
     * @return some value
     */
    double weightDelta(double grad, double delta);

    /**
     * Gradient for neuron
     * @param out - actual output
     * @return delta for outputs
     */
    double gradient(double out, double delta);

    /**
     * Delta for neuron:
     * @param out - OUT
     * @param weight - Wi
     * @param delta - OUT.delta
     * @return delta for outputs
     */
    double neuronDelta(double out,
                       double weight,
                       double delta);

    /**
     * Delta for outputs:
     * @param actual - actual value
     * @param ideal - ideal value
     * @return delta for outputs
     */
    double outputDelta(double actual, double ideal);
}