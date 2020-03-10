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

import static java.util.Arrays.fill;

/**
 * SimpleVertices  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class IoLayer implements Layer {

    private final double[] net;
    private final double[] out;
    private final int size;

    public IoLayer(final int size) {

        this.size = size;
        net = new double[size];
        out = new double[size];
        fill(net, 0.);
        fill(out, 0.);
    }

    @Override
    public void clean() {
        fill(net, 0.);
        fill(out, 0.);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public double net(final int i) {
        return net[i];
    }

    @Override
    public void net(final int i, final double v) {
        net[i] = v;
    }

    @Override
    public double out(final int i) {
        return out[i];
    }

    @Override
    public void out(final int i, final double v) {
        out[i] = v;
    }
}