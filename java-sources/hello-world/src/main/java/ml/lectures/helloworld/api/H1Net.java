package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.Consumer;

/**
 * OneLayerMachine
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class H1Net implements LNet {

    private final LMath math;

    /**
     * Constructor
     * @param math - math
     */
    public H1Net(final LMath math) {
        this.math = math;
    }

    public void check(final Weights weights, final TrainSet set, Consumer<Double> error) {

        val layers = new H1Layers(weights.isize(), weights.hsize(), weights.osize());
        set.forEach(
            (d, t) -> {
                layers.clean();
                forward(d, weights, layers);
                for (int i = 0; i < layers.olayer().size(); i++) {
                    error.accept(
                        math.deviation(layers.olayer().out(i), t[i])
                    );
                }
            }
        );
    }

    @Override
    public void train(final Weights weights, final TrainSet set) {

        val deltas = new ArrayWeights(weights.isize(), weights.hsize(), weights.osize());
        val layers = new H1Layers(weights.isize(), weights.hsize(), weights.osize());
        set.forEach(
            (d, t) -> {
                forward(d, weights, layers);
                backward(layers, weights, deltas, t);
            }
        );
        fixWeights(weights, deltas);
    }

//    private void clear(final Layer[] layers) {
//        layers[I_LAYER].clean();
//        layers[H_LAYER].clean();
//        layers[B_LAYER].clean();
//        layers[O_LAYER].clean();
//    }

    private void fixWeights(final Weights weights, final ArrayWeights deltas) {

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

    private void forward(final double[] set,
                         final Weights weights,
                         final Layers layers) {

        layers.clean();
        val il = layers.ilayer();
        val hl = layers.hlayer();
        val bl = layers.blayer();
        val ol = layers.olayer();

        for(int i = 0; i < il.size(); i++) {
            il.net(i, set[i]);
        }

        for (int i = 0; i < il.size(); i++) {
            for (int j = 0; j < hl.size(); j++) {
                hl.net(j, hl.net(j) + il.out(i) * weights.i2h(i, j));
            }
        }

        for (int i = 0; i < bl.size(); i++) {
            for (int j = 0; j < hl.size(); j++) {
                hl.net(j, hl.net(j) + bl.out(i) * weights.b2h(i, j));
            }
        }

        for (int i = 0; i < hl.size(); i++) {
            hl.out(i, math.activation(hl.net(i)));
        }

        for (int i = 0; i < ol.size(); i++) {
            double net = 0.;
            for (int j = 0; j < hl.size(); j++) {
                net += hl.out(j) * weights.h2o(j, i);
            }
            ol.net(i, net);
            ol.out(i, math.activation(net));
        }
    }

    private void backward(final Layers layers,
                          final Weights weights,
                          final Weights deltas,
                          final double[] target) {

        val il = layers.ilayer();
        val hl = layers.hlayer();
        val bl = layers.blayer();
        val ol = layers.olayer();

        val odeltas = new double[deltas.osize()];
        for (int i = 0; i < deltas.osize(); i++) {
            odeltas[i] = math.doutput(ol.out(i), target[i]);
        }

        for (int i = 0; i < deltas.osize(); i++) {
            for (int j = 0; j < deltas.hsize(); j++) {
                deltas.h2o(j, i,
                    math.dweight(
                        math.gradient(
                            hl.out(j),
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
            hdeltas[i] = math.dneuron(hl.out(i), ws, ds);
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.isize(); j++) {
                deltas.i2h(j, i,
                    math.dweight(math.gradient(il.out(j), hdeltas[i]), deltas.i2h(j, i))
                );
            }
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.bsize(); j++) {
                deltas.b2h(j, i,
                    math.dweight(math.gradient(bl.out(j), hdeltas[i]), deltas.b2h(j, i))
                );
            }
        }
    }
}
