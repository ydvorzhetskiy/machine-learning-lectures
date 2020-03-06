package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.Consumer;

/**
 * OneLayerMachine
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class OneLayerNet implements LNet {

    private static final int I_VERT = 0;
    private static final int H_VERT = 1;
    private static final int B_VERT = 2;
    private static final int O_VERT = 3;
    
    private final LMath math;

    /**
     * Constructor
     * @param math - math
     */
    public OneLayerNet(final LMath math) {
        this.math = math;
    }

    public void check(final Weights weights, final TrainSet set, Consumer<Double> error) {
        val vertices = new Vertices[O_VERT + 1];
        vertices[I_VERT] = new SimpleVertices(weights.isize());
        vertices[H_VERT] = new SimpleVertices(weights.hsize());
        vertices[B_VERT] = new SimpleVertices(weights.bsize());
        vertices[O_VERT] = new SimpleVertices(weights.osize());
        set.forEach(
            (d, t) -> {
                forward(d, weights, vertices);
                for (int i = 0; i < vertices[O_VERT].size(); i++) {
                    error.accept(
                        math.deviation(vertices[O_VERT].net(i), t[i])
                    );
                }
            }
        );

    }

    @Override
    public void train(final Weights weights, final TrainSet set) {

        val deltas = new SimpleWeights(weights.isize(), weights.hsize(), weights.bsize(), weights.osize());
        val vertices = new Vertices[O_VERT + 1];
        vertices[I_VERT] = new SimpleVertices(weights.isize());
        vertices[H_VERT] = new SimpleVertices(weights.hsize());
        vertices[B_VERT] = new SimpleVertices(weights.bsize());
        vertices[O_VERT] = new SimpleVertices(weights.osize());
        set.forEach(
            (d, t) -> {
                forward(d, weights, vertices);
                backward(vertices, weights, deltas, t);
            }
        );
        fixWeights(weights, deltas);
    }

    private void fixWeights(final Weights weights, final SimpleWeights deltas) {
        for (int i = 0; i < weights.hsize(); i++) {
            for (int j = 0; j < weights.isize(); j++) {
                weights.i2h(j, i, weights.i2h(j, i) + deltas.i2h(j, i));
            }
            for (int j = 0; j < weights.bsize(); j++) {
                weights.b2h(j, i, weights.b2h(j, i) + deltas.b2h(j, i));
            }
            for (int j = 0; j < weights.osize(); j++) {
                weights.h2o(i, j, weights.h2o(i, j) + deltas.h2o(i, j));
            }
        }
    }

    private void clean(final Vertices[] vertices) {

        val ivert = vertices[I_VERT];
        val hvert = vertices[H_VERT];
        val overt = vertices[O_VERT];

        for (int i = 0; i < overt.size(); i++) {
            overt.net(i, 0.);
        }
        for (int i = 0; i < hvert.size(); i++) {
            hvert.net(i, 0.);
            hvert.out(i, 0.);
        }
        for (int i = 0; i < ivert.size(); i++) {
            ivert.out(i, 0.);
        }
    }

    private void forward(final double[] set,
                         final Weights weights,
                         final Vertices[] vertices) {

        clean(vertices);
        val ivert = vertices[I_VERT];
        val hvert = vertices[H_VERT];
        val bvert = vertices[B_VERT];
        val overt = vertices[O_VERT];

        init(set, vertices, weights);

        for (int i = 0; i < hvert.size(); i++) {
            double net = 0.;

            for (int j = 0; j < ivert.size(); j++) {
                net += ivert.out(j) * weights.i2h(j, i);
            }

            for (int j = 0; j < bvert.size(); j++) {
                net += bvert.out(j) * weights.b2h(j, i);
            }
            hvert.out(i, hvert.out(i) + math.logisticFun(net));

            for (int j = 0; j < overt.size(); j++) {
                overt.net(j, overt.net(j) + hvert.out(i) * weights.h2o(i, j));
            }
        }
    }

    private void init(final double[] set,
                      final Vertices[] vertices,
                      final Weights weights) {

        val ivert = vertices[I_VERT];
        val bvert = vertices[B_VERT];

        for(int i = 0; i < ivert.size(); i++) {
            ivert.out(i, set[i]);
        }
        for (int i = 0; i < weights.bsize(); i++) {
            bvert.out(i, 1);
        }
    }

    private void backward(final Vertices[] vertices,
                          final Weights weights,
                          final Weights deltas,
                          final double[] target) {

        val ivert = vertices[I_VERT];
        val hvert = vertices[H_VERT];
        val bvert = vertices[B_VERT];
        val overt = vertices[O_VERT];

        val odeltas = new double[deltas.osize()];
        for (int i = 0; i < deltas.osize(); i++) {
            odeltas[i] = math.outputDelta(overt.net(i), target[i]);
        }

        for (int i = 0; i < deltas.osize(); i++) {
            for (int j = 0; j < deltas.hsize(); j++) {
                deltas.h2o(j, i,
                    math.weightDelta(
                        math.gradient(
                            hvert.out(j),
                            odeltas[i]
                        ),
                        deltas.h2o(j, i)
                    )
                );
            }
        }

        val hdeltas = new double[deltas.hsize()];
        for (int i = 0; i < deltas.hsize(); i++) {
            val ws = new double[deltas.osize()];
            val ds = new double[deltas.osize()];
            for (int j = 0; j < deltas.osize(); j++) {
                ws[j] = weights.h2o(i, j);
                ds[j] = odeltas[j];
            }
            hdeltas[i] = math.neuronDelta(hvert.out(i), ws, ds);
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.isize(); j++) {
                deltas.i2h(j, i,
                    math.weightDelta(math.gradient(ivert.out(j), hdeltas[i]), deltas.i2h(j, i))
                );
            }
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.bsize(); j++) {
                deltas.b2h(j, i,
                    math.weightDelta(math.gradient(bvert.out(j), hdeltas[i]), deltas.b2h(j, i))
                );
            }
        }
    }
}
