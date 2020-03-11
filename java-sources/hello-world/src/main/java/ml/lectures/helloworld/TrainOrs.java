package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.SigmoidMath;
import ml.lectures.helloworld.api.TrainSet;

import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.api.Utils.or;

/**
 * TrainOrs  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainOrs {

    public static void main(String[] args) {

        val net = new H1Net(new SigmoidMath(1.8, 0.8));
        val weights = TrainCommon.weights();
        final TrainSet set = TrainCommon.trainSet(
            inputs -> or(inputs[0], inputs[1])
        );

        TrainCommon.train(net, BPOINTS, set, weights, 10000);
    }
}