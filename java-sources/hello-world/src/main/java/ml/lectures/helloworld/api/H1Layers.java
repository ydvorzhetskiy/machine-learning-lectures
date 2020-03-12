package ml.lectures.helloworld.api;

import java.util.function.Function;

/**
 * H1Layers  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class H1Layers implements Layers {

    private final Layer ilayer;
    private final Layer hlayer;
    private final Layer blayer;
    private final Layer olayer;


    public H1Layers(final int isize,
                    final int hsize,
                    final int osize,
                    final Function<Double, Double> activationFun) {

        ilayer = new Sensors(isize);
        hlayer = new ActiveLayer(hsize, activationFun);
        blayer = new Biases();
        olayer = new ActiveLayer(osize, activationFun);
    }

    @Override
    public Layer ilayer() {
        return ilayer;
    }

    @Override
    public Layer hlayer() {
        return hlayer;
    }

    @Override
    public Layer blayer() {
        return blayer;
    }

    @Override
    public Layer olayer() {
        return olayer;
    }

    @Override
    public void clean() {
        ilayer.clean();
        hlayer.clean();
        blayer.clean();
        olayer.clean();
    }
}