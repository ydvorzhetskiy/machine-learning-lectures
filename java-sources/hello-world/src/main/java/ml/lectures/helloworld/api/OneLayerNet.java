package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.Consumer;

/**
 * OneLayerMachine
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class OneLayerNet implements LNet {

    private static final int I_LAYER = 0;
    private static final int H_LAYER = 1;
    private static final int B_LAYER = 2;
    private static final int O_LAYER = 3;
    
    private final LMath math;

    /**
     * Constructor
     * @param math - math
     */
    public OneLayerNet(final LMath math) {
        this.math = math;
    }

    public void check(final Weights weights, final TrainSet set, Consumer<Double> error) {
        val layers = layers(weights);
        set.forEach(
            (d, t) -> {
                clear(layers);
                forward(d, weights, layers);
                for (int i = 0; i < layers[O_LAYER].size(); i++) {
                    error.accept(
                        math.deviation(layers[O_LAYER].net(i), t[i])
                    );
                }
            }
        );

    }

    @Override
    public void train(final Weights weights, final TrainSet set) {

        val deltas = new SimpleWeights(weights.isize(), weights.hsize(), weights.bsize(), weights.osize());
        val layers = layers(weights);
        set.forEach(
            (d, t) -> {
                clear(layers);
                forward(d, weights, layers);
                backward(layers, weights, deltas, t);
            }
        );
        fixWeights(weights, deltas);
    }

    private void clear(final Layer[] layers) {
        layers[I_LAYER].clean();
        layers[H_LAYER].clean();
        layers[B_LAYER].clean();
        layers[O_LAYER].clean();
    }

    private Layer[] layers(final Weights weights) {
        val layers = new Layer[O_LAYER + 1];
        layers[I_LAYER] = new UnoLayer(weights.isize());
        layers[H_LAYER] = new IoLayer(weights.hsize());
        layers[B_LAYER] = new BLayer();
        layers[O_LAYER] = new UnoLayer(weights.osize());
        return layers;
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

    private void clean(final Layer[] layers) {

        val il = layers[I_LAYER];
        val hl = layers[H_LAYER];
        val ol = layers[O_LAYER];

        for (int i = 0; i < ol.size(); i++) {
            ol.net(i, 0.);
        }
        for (int i = 0; i < hl.size(); i++) {
            hl.net(i, 0.);
            hl.out(i, 0.);
        }
        for (int i = 0; i < il.size(); i++) {
            il.out(i, 0.);
        }
    }

    private void forward(final double[] set,
                         final Weights weights,
                         final Layer[] layers) {

        clean(layers);
        val il = layers[I_LAYER];
        val hl = layers[H_LAYER];
        val bl = layers[B_LAYER];
        val ol = layers[O_LAYER];

        init(set, layers);

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
            hl.out(i, math.logisticFun(hl.net(i)));
        }

        for (int i = 0; i < ol.size(); i++) {
            double net = 0.;
            for (int j = 0; j < hl.size(); j++) {
                net += hl.out(j) * weights.h2o(j, i);
            }
            ol.net(i, net);
        }
    }

    private void init(final double[] set,
                      final Layer[] layers) {

        val ih = layers[I_LAYER];
        for(int i = 0; i < ih.size(); i++) {
            ih.out(i, set[i]);
        }
    }

    private void backward(final Layer[] layers,
                          final Weights weights,
                          final Weights deltas,
                          final double[] target) {

        val il = layers[I_LAYER];
        val hl = layers[H_LAYER];
        val bl = layers[B_LAYER];
        val ol = layers[O_LAYER];

        val odeltas = new double[deltas.osize()];
        for (int i = 0; i < deltas.osize(); i++) {
            odeltas[i] = math.outputDelta(ol.net(i), target[i]);
        }

        for (int i = 0; i < deltas.osize(); i++) {
            for (int j = 0; j < deltas.hsize(); j++) {
                deltas.h2o(j, i,
                    math.weightDelta(
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
            hdeltas[i] = math.neuronDelta(hl.out(i), ws, ds);
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.isize(); j++) {
                deltas.i2h(j, i,
                    math.weightDelta(math.gradient(il.out(j), hdeltas[i]), deltas.i2h(j, i))
                );
            }
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.bsize(); j++) {
                deltas.b2h(j, i,
                    math.weightDelta(math.gradient(bl.out(j), hdeltas[i]), deltas.b2h(j, i))
                );
            }
        }
    }
}
