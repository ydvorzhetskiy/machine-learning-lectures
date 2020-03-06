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
 * Vertices  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface Vertices {

    /**
     * Gets size
     * @return size
     */
    int size();

    /**
     * Get vertex net (input) value
     * @param i - vertex index
     * @return vertex net (input) value
     */
    double net(final int i);

    /**
     * Set vertex net (input) value
     * @param i - vertex index
     * @param v - vertex net (input) value
     */
    void net(final int i, double v);

    /**
     * Get vertex out value
     * @param i - vertex index
     * @return vertex out value
     */
    double out(final int i);

    /**
     * Set vertex out value
     * @param i - vertex index
     * @param v - vertex out value
     */
    void out(final int i, double v);
}