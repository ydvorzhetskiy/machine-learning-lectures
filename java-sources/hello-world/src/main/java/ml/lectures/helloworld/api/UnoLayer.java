package ml.lectures.helloworld.api;

import static java.util.Arrays.fill;

/**
 * UnoLayer  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class UnoLayer implements Layer {

    private final double[] val;
    private final int size;

    public UnoLayer(final int size) {

        this.size = size;
        val = new double[size];
        fill(val, 0.);
    }

    @Override
    public void clean() {
        fill(val, 0.);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public double net(final int i) {
        return val[i];
    }

    @Override
    public void net(final int i, final double v) {
        val[i] = v;
    }

    @Override
    public double out(final int i) {
        return val[i];
    }

    @Override
    public void out(final int i, final double v) {
        val[i] = v;
    }
}