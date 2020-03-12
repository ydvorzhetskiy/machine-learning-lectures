package ml.lectures.helloworld.api;

/**
 * BLayer  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class Biases implements Layer {

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

}