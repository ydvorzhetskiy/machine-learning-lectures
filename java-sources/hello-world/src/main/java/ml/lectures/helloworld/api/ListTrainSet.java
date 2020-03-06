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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * ArrayTrainSet  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class ListTrainSet implements TrainSet {

    private final List<double[]> list = new ArrayList<>();

    @Override
    public TrainSet add(double[] set) {
        list.add(set);
        return this;
    }

    @Override
    public void forEach(final Consumer<double[]> consumer) {
        list.forEach(consumer);
    }
}