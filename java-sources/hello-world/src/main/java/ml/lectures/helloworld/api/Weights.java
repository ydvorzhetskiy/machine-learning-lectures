package ml.lectures.helloworld.api;

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

    Weights b2h(int b, int h, double v);

    double b2h(int b, int h);

    Weights h2o(int h, int o, double v);

    double h2o(int h, int o);
}