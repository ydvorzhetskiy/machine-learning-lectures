package ml.lectures.helloworld.api;

import lombok.val;

/**
 * Edges
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface Weights {

    int isize();

    int hsize();

    int bsize();

    int osize();

    Weights i2h(int i, int h, double v);

    double i2h(int i, int h);

    default Weights i2h(int i, double[] v) {
        for (int h = 0; h < hsize(); h++) {
            i2h(i, h, v[h]);
        }
        return this;
    }

    default double[] i2h(int i) {
        val ret = new double[hsize()];
        for (int h = 0; h < hsize(); h++) {
            ret[i] = i2h(i, h);
        }
        return ret;
    }

    Weights b2h(int b, int h, double v);

    double b2h(int b, int h);

    default Weights b2h(int b, double[] v) {
        for (int i = 0; i < hsize(); i++) {
            b2h(b, i, v[i]);
        }
        return this;
    }

    default double[] b2h(int b) {
        val ret = new double[hsize()];
        for (int i = 0; i < hsize(); i++) {
            ret[i] = b2h(b, i);
        }
        return ret;
    }

    Weights h2o(int h, int o, double v);

    double h2o(int h, int o);

    default Weights h2o(int h, double[] v) {
        for (int i = 0; i < osize(); i++) {
            h2o(h, i, v[i]);
        }
        return this;
    }

    default double[] h2o(int h) {
        val ret = new double[osize()];
        for (int i = 0; i < osize(); i++) {
            ret[i] = h2o(h, i);
        }
        return ret;
    }
}