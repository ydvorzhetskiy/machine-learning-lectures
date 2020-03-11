package ml.lectures.helloworld.api;

import static java.util.Arrays.fill;

/**
 * SimpleEdges  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class ArrayWeights implements Weights {

    private final double[] i2h;
    private final double[] h2o;
    private final double[] b2h;
    private final int isize;
    private final int hsize;
    private final int bsize;
    private final int osize;

    public ArrayWeights(final int isize,
                        final int hsize,
                        final int osize) {

        this.isize = isize;
        this.hsize = hsize;
        this.bsize = 1;
        this.osize = osize;
        i2h = new double[isize * hsize];
        h2o = new double[osize * hsize];
        b2h = new double[bsize * hsize];
        fill(i2h, 0.);
        fill(h2o, 0.);
        fill(b2h, 0.);
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
        i2h[offset(i, h)] = v;
        return this;
    }

    private int offset(final int i, final int h) {
        return i * hsize + h;
    }

    @Override
    public double i2h(int i, int h) {
        return i2h[offset(i, h)];
    }

    @Override
    public Weights b2h(int b, int h, double v) {
        b2h[offset(b, h)] = v;
        return this;
    }

    @Override
    public double b2h(int b, int h) {
        return b2h[offset(b, h)];
    }

    @Override
    public Weights h2o(int h, int o, double v) {
        h2o[offset(o, h)] = v;
        return this;
    }

    @Override
    public double h2o(int h, int o) {
        return  h2o[offset(o, h)];
    }
}
