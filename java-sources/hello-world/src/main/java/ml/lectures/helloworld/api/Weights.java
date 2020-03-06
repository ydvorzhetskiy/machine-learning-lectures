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

/**
 * Edges
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface Weights {

    void dump();

    int isize();

    int hsize();

    int bsize();

    int osize();

    Weights i2h(int i, int h, double v);

    double i2h(int i, int h);

    Weights b2h(int b, int h, double v);

    double b2h(int b, int h);

    Weights h2o(int h, int o, double v);

    double h2o(int h, int o);
}