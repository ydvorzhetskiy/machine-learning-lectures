/*
 *  Copyright 2020 Russian Post
 *
 * This source code is Russian Post Confidential Proprietary.
 * This software is protected by copyright. All rights and titles are reserved.
 * You shall not use, copy, distribute, modify, decompile, disassemble or reverse engineer the software.
 * Otherwise this violation would be treated by law and would be subject to legal prosecution.
 * Legal use of the software provides receipt of a license from the right holder only.
 */
package ml.lectures.helloworld.api;

import java.util.function.BiConsumer;

/**
 * TrainSet  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface TrainSet {

    /**
     * forEach
     * @param consumer - consumer
     */
    void forEach(BiConsumer<double[], double[]> consumer);

    /**
     * add
     * @param data - set
     * @param target - target
     * @return this
     */
    TrainSet add(double[] data, double[] target);
}