package ml.lectures.helloworld.api;

import lombok.val;

/**
 * OneLayerMachine
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class OneLayerNet implements LNet {

    private final LMath math;

    /**
     * Constructor
     * @param math - math
     */
    public OneLayerNet(final LMath math) {
        this.math = math;
    }

    @Override
    public void train(final Weights weights, final TrainSet set, double[] target) {

        val deltas = new SimpleWeights(weights.isize(), weights.hsize(), weights.bsize(), weights.osize());
        val vertices = new Vertices[4];
        vertices[0] = new SimpleVertices(weights.isize());
        vertices[1] = new SimpleVertices(weights.hsize());
        vertices[2] = new SimpleVertices(weights.bsize());
        vertices[3] = new SimpleVertices(weights.osize());
        clean(vertices);
        set.forEach(data -> forward(data, weights, vertices));
        backward(vertices, weights, deltas, target);
    }

    private void clean(final Vertices[] vertices) {

        val ivert = vertices[0];
        val hvert = vertices[1];
        val overt = vertices[3];

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

        val ivert = vertices[0];
        val hvert = vertices[1];
        val bvert = vertices[2];
        val overt = vertices[3];

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

        val ivert = vertices[0];
        val bvert = vertices[2];

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

        val ivert = vertices[0];
        val hvert = vertices[1];
        val bvert = vertices[2]; //why not used?
        val overt = vertices[3];

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