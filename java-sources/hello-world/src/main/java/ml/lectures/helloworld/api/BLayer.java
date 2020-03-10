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
 * BLayer  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class BLayer implements Layer {

    @Override
    public void clean() {
        //nop
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public double net(final int i) {
        return 1;
    }

    @Override
    public void net(final int i, final double v) {
        //nop
    }

    @Override
    public double out(final int i) {
        return 1;
    }

    @Override
    public void out(final int i, final double v) {
        //nop
    }
}