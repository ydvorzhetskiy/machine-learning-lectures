package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.Function;

import static ml.lectures.helloworld.api.Utils.add;
import static ml.lectures.helloworld.api.Utils.join;
import static ml.lectures.helloworld.api.Utils.mult;
import static ml.lectures.helloworld.api.Utils.operate;
import static ml.lectures.helloworld.api.Utils.sum;

/**
 * OneLayerMachine
 *
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
        val bl = layers.blayer();
        val ol = layers.olayer();
        il.net(set);

        hl.net(
            add(mult(il.out(), weights.i2h()),
                mult(bl.out(), weights.b2h()))
        );

        ol.net(mult(hl.out(), weights.h2o()));
    }

    private void backward(final Layers layers,
                          final Weights weights,
                          final Weights deltas,
                          final double[] target) {

        val il = layers.ilayer();
        val hl = layers.hlayer();
        val bl = layers.blayer();
        val ol = layers.olayer();

        val errs = operate(ol.out(), target, math::odelta);
        deltas.h2o(
            operate(
                join(hl.out(), errs, math::grad),
                deltas.h2o(),
                math::dweight
            )
        );

        val hdeltas = new double[deltas.hsize()];
        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.osize(); j++) {
                hdeltas[i] += math.hdelta(hl.out(i), weights.h2o(i, j), errs[j]);
            }
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.isize(); j++) {
                deltas.i2h(j, i,
                    math.dweight(math.grad(il.out(j), hdeltas[i]), deltas.i2h(j, i))
                );
            }
        }

        for (int i = 0; i < deltas.hsize(); i++) {
            for (int j = 0; j < deltas.bsize(); j++) {
                deltas.b2h(j, i,
                    math.dweight(math.grad(bl.out(j), hdeltas[i]), deltas.b2h(j, i))
                );
            }
        }
    }

    /*

    Mind.prototype.back = function(examples, results) {
  var activatePrime = this.activatePrime;
  var learningRate = this.learningRate;
  var weights = this.weights;

  // compute weight adjustments
  var errorOutputLayer = subtract(examples.output, results.outputResult);
  var deltaOutputLayer = dot(results.outputSum.transform(activatePrime), errorOutputLayer);
  var hiddenOutputChanges = scalar(multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
  var deltaHiddenLayer = dot(multiply(weights.hiddenOutput.transpose(), deltaOutputLayer), results.hiddenSum.transform(activatePrime));
  var inputHiddenChanges = scalar(multiply(deltaHiddenLayer, examples.input.transpose()), learningRate);

  // adjust weights
  weights.inputHidden = add(weights.inputHidden, inputHiddenChanges);
  weights.hiddenOutput = add(weights.hiddenOutput, hiddenOutputChanges);

  return errorOutputLayer;
};
    */
}
