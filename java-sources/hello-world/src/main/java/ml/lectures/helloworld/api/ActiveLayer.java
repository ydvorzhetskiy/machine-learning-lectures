package ml.lectures.helloworld.api;

import java.util.function.Function;

import static java.util.Arrays.fill;

/**
 * SimpleVertices  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class ActiveLayer implements Layer {

    private final double[] net;
    private final double[] out;
    private final int size;
    private final Function<Double, Double> activationFun;

    public ActiveLayer(final int size,
                       final Function<Double, Double> activationFun) {

        this.size = size;
        net = new double[size];
        out = new double[size];
        this.activationFun = activationFun;
        clean();
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
        out[i] = this.activationFun.apply(v);
    }

    @Override
    public double out(final int i) {
        return out[i];
    }

}