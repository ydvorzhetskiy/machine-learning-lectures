package ml.lectures.helloworld;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import lombok.val;

import static java.lang.System.arraycopy;
import static java.util.Arrays.fill;

/**
 * OneHiddenLayerMachine  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
@Accessors(chain = true)
@Getter
@Setter
public class OneLayerMachine implements LMachine {

    private LearnMath math;

    private double[] weights;
    private int hsize;

    private int[] iws;
    private int[] bws;
    private int[] ows;

    @Override
    public void train(final double[][] set,
                      final double[] ideal) {

        val deltas = new double[wsize()];

        for (int i = 0; i < set.length; i++) {
            val outputs = new double[vsize()];
            fill(outputs, 0.);

            forward(set[i], weights, outputs);
            backward(weights, deltas, outputs, ideal[i]);
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i] + deltas[i];
        }
    }

    @Override
    public int vsize() {

        return iws[1] + hsize + bws[1] + ows[1];
    }

    @Override
    public int wsize() {

        return iws[1] * hsize + bws[1] * hsize + ows[1] * hsize;
    }

    private void forward(final double[] set,
                         final double[] weights,
                         final double[] outputs) {

        initOutputs(set, outputs);

        for (int i = 0; i < hsize; i++) {
            double net = 0.;
            //calc h-neuron nets from i-acceptors
            for (int j = iws[0]; j < iws[1]; j++) {
                net += outputs[j] * weights[ihw(j, i)];
            }
            //calc h-neuron nets from b-neurons (bias)
            for (int j = bws[0]; j < bws[1]; j++) {
                net += outputs[j] * weights[bhw(j, i)];
            }
            outputs[i] = math.logisticFun(net);

            for (int j = ows[0]; j < ows[1]; i++) {
                outputs[j] = outputs[j] + outputs[i] * weights[how(i, j)];
            }
        }
    }

    /**
     * index of weight between hidden neuron and output
     * @param i - h-index
     * @param j - o-index
     * @return weight between input and hidden neuron
     */
    int how(final int i, final int j) {

        return (iws[1] + bws[1] + j) * hsize + i;
    }

    /**
     * gets weight between bias and hidden neuron
     * @param i - b-index
     * @param j - h-index
     * @return weight between input and hidden neuron
     */
    int bhw(final int i, final int j) {

        return (iws[1] + i) * hsize + j;
    }

    /**
     * gets weight between input and hidden neuron
     * @param i - i-index
     * @param j - h-index
     * @return weight between input and hidden neuron
     */
    int ihw(final int i, final int j) {

        return hsize * i + j;
    }

    /**
     * Fill vertexes's output array with known values from input and bias
     * @param set - train set
     * @param outputs - outputs
     */
    private void initOutputs(final double[] set, final double[] outputs) {


        arraycopy(set, iws[0], outputs, iws[0], iws[1] - iws[0]);

        for (int i = bws[0]; i < bws[1]; i++) {
            outputs[i] = 1;
        }
    }

    private void backward(final double[] weights,
                          final double[] deltaws,
                          final double[] outputs,
                          final double ideal) {

    }
}