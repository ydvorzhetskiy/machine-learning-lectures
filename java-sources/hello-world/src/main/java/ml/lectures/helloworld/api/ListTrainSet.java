package ml.lectures.helloworld.api;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

/**
 * ArrayTrainSet  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class ListTrainSet implements TrainSet {

    private static class D {
        final double[] data;
        final double[] target;

        public D(final double[] data, final double[] target) {
            this.data = data;
            this.target = target;
        }
    }
    private final List<D> list = new ArrayList<>();

    @Override
    public TrainSet add(double[] data, double[] target) {
        list.add(new D(data, target));
        return this;
    }

    @Override
    public void forEach(final BiConsumer<double[], double[]> consumer) {
        list.forEach(
            d -> consumer.accept(d.data, d.target)
        );
    }
}