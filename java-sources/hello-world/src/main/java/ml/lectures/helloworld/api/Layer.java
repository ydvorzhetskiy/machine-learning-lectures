package ml.lectures.helloworld.api;

import lombok.val;

/**
 * Vertices  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface Layer {

    /**
     * clean data
     */
    void clean();

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

    default double[] out() {
        val result = new double[size()];
        for (int i = 0; i < size(); i++) {
            result[i] = out(i);
        }
        return result;
    }

    default double[] net() {
        val result = new double[size()];
        for (int i = 0; i < size(); i++) {
            result[i] = net(i);
        }
        return result;
    }

    default void net(double[] arr) {
        for (int i = 0; i < size(); i++) {
            net(i, arr[i]);
        }
    }

    default void out(double[] arr) {
        for (int i = 0; i < size(); i++) {
            out(i, arr[i]);
        }
    }
}