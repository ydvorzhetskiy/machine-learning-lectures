package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.Function;

import static ml.lectures.helloworld.api.Utils.column;
import static ml.lectures.helloworld.api.Utils.mul;
import static ml.lectures.helloworld.api.Utils.oper;
import static ml.lectures.helloworld.api.Utils.row;
import static ml.lectures.helloworld.api.Utils.sum;
import static ml.lectures.helloworld.api.Utils.tran;

/**
 * OneLayerMachine
 * Expected results for xors
 * 6.008	0.771	-5.498	1.527	2.296	1.378	-8.862	4.210
 * epoch: 5000	error:	0.000
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class H1Net implements LNet {

    private final LMath math;
    private final Function<Double, Double> activationFun;

    /**
     * Constructor
     * @param math - math
     */
    public H1Net(final LMath math) {
        this.math = math;
        activationFun = H1Net.this.math::activation;
    }

    public double[] test(final Weights weights, final double[] input) {

        val layers = newLayers(weights);
        forward(input, weights, layers);
        return layers.olayer().out();
    }

    private H1Layers newLayers(final Weights weights) {
        return new H1Layers(weights.isize(),
            weights.hsize(),
            weights.osize(),
            activationFun);
    }

    @Override
    public void train(final Weights weights, final TrainSet set) {

        val deltas = new ArrayWeights(weights.isize(), weights.hsize(), weights.osize());
        val layers = newLayers(weights);
        set.forEach(
            (d, t) -> {
                forward(d, weights, layers);
                backward(layers, weights, deltas, t);
            }
        );
        fixWeights(weights, deltas);
    }

    private void fixWeights(final Weights weights, final ArrayWeights deltas) {

        weights.i2h(sum(weights.i2h(), deltas.i2h()));
        weights.b2h(sum(weights.b2h(), deltas.b2h()));
        weights.h2o(sum(weights.h2o(), deltas.h2o()));
    }

    private void forward(final double[] set,
                         final Weights weights,
                         final Layers layers) {

        layers.clean();
        val il = layers.ilayer();
        val hl = layers.hlayer();
        val ol = layers.olayer();

        il.net(set);

        val hout = row(hl.out());
        val iout = row(il.out());
        val bout = row(layers.blayer().out());

        hl.net(row(sum(mul(iout, weights.i2h()), mul(bout, weights.b2h())),0));

        ol.net(row(mul(hout, weights.h2o()),0));
    }

    private void backward(final Layers layers,
                          final Weights weights,
                          final Weights deltas,
                          final double[] target) {

        val hout = column(layers.hlayer().out());
        val iout = column(layers.ilayer().out());
        val bout = column(layers.blayer().out());
        val oout = column(layers.olayer().out());

        val errs = oper(oout, column(target), math::odelta);

        deltas.h2o(oper(mul(hout, errs), deltas.h2o(), math::dweight));
        val dhs = tran(oper(hout, mul(weights.h2o(), errs), math::hdelta));
        deltas.i2h(oper(mul(iout, dhs), deltas.i2h(), math::dweight));
        deltas.b2h(oper(mul(bout, dhs), deltas.b2h(), math::dweight));
    }
}
