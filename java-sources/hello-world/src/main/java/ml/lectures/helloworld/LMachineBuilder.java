package ml.lectures.helloworld;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * MlBuilder
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
@Accessors(chain = true)
@Getter
@Setter
public class LMachineBuilder {

    private LearnMath math;
    private int inputs;
    private int outputs;
    private int neyrons;
    private double[] initialWeights;

    public LMachine build() {
        return null;
    }
}