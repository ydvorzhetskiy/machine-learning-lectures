package ml.lectures.helloworld.api;

/**
 * Layers  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public interface Layers {
    Layer ilayer();
    Layer hlayer();
    Layer blayer();
    Layer olayer();

    void clean();
}