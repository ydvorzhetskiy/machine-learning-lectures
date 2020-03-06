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

import lombok.val;

import static java.util.Arrays.fill;

/**
 * SimpleEdges  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class SimpleWeights implements Weights {

    private final double[] i2h;
    private final double[] h2o;
    private final double[] b2h;
    private final int isize;
    private final int hsize;
    private final int bsize;
    private final int osize;

    public SimpleWeights(final int isize,
                         final int hsize,
                         final int bsize,
                         final int osize) {

        this.isize = isize;
        this.hsize = hsize;
        this.bsize = bsize;
        this.osize = osize;
        i2h = new double[isize * hsize];
        h2o = new double[osize * hsize];
        b2h = new double[bsize * hsize];
        fill(i2h, 0.);
        fill(h2o, 0.);
        fill(b2h, 0.);
    }

    @Override
    public void dump() {
        val out = new StringBuilder();
        for (int i = 0; i < this.isize; i++) {
            for (int j = 0; j < this.hsize; j++) {
                out.append(String.format("\t%.3f", i2h(i, j)));
            }
        }

        for (int i = 0; i < this.bsize; i++) {
            for (int j = 0; j < this.hsize; j++) {
                out.append(String.format("\t%.3f", b2h(i, j)));
            }
        }

        for (int i = 0; i < this.osize; i++) {
            for (int j = 0; j < this.hsize; j++) {
                out.append(String.format("\t%.3f", h2o(j, i)));
            }
        }

        System.out.println(out);
    }

    @Override
    public int isize() {
        return isize;
    }

    @Override
    public int hsize() {
        return hsize;
    }

    @Override
    public int bsize() {
        return bsize;
    }

    @Override
    public int osize() {
        return osize;
    }

    @Override
    public Weights i2h(int i, int h, double v) {
        i2h[i * hsize + h] = v;
        return this;
    }

    @Override
    public double i2h(int i, int h) {
        return i2h[i * hsize + h];
    }

    @Override
    public Weights b2h(int b, int h, double v) {
        b2h[b * hsize + h] = v;
        return this;
    }

    @Override
    public double b2h(int b, int h) {
        return b2h[b * hsize + h];
    }

    @Override
    public Weights h2o(int h, int o, double v) {
        h2o[o * hsize + h] = v;
        return this;
    }

    @Override
    public double h2o(int h, int o) {
        return  h2o[o * hsize + h];
    }
}
