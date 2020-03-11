package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.ArrayWeights;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.SigmoidMath;
import ml.lectures.helloworld.api.TrainSet;

import static ml.lectures.helloworld.api.Utils.or;

/**
 * LearnOrs  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainI2H3O1Ors {

    public static void main(String[] args) {

//        val net = new H1Net(new SigmoidMath(0.9, 0.5));
        val net = new H1Net(new SigmoidMath(0.9, 0.5));
        val weights = new ArrayWeights(2, 3, 1)
            .i2h(0, 0, 0.5)
            .i2h(0, 1, 0.3)
            .i2h(1, 0, -0.5)
            .i2h(1, 1, 0.5)
            .i2h(1, 2, 0.5)
            .h2o(0, 0, 0.2)
            .h2o(1, 0, 0.3)
            .h2o(2, 0, 0.4)
            .b2h(0, 0, -0.2)
            .b2h(0, 1, 0.2)
            .b2h(0, 2, 0.2);

        final TrainSet set = TrainCommon.trainSet(
            inputs -> or(inputs[0], inputs[1])
        );
        TrainCommon.train(
            net,
            new int[] {1, 100, 500, 1_000, 2_500, 5_000, 10_000},
            set,
            weights,
            10_000
        );
    }
}