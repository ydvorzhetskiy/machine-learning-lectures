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