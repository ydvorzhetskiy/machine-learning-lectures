package ml.lectures.helloworld.api;

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


    public H1Layers(final int isize, final int hsize, final int osize) {
        ilayer = new UnoLayer(isize);
        hlayer = new IoLayer(hsize);
        blayer = new BLayer();
        olayer = new IoLayer(osize);
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