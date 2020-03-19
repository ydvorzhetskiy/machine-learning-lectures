package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.SigmoidMath;

import java.util.function.Function;

import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.TrainCommon.trainSet;
import static ml.lectures.helloworld.api.Utils.xor;

/**
 * LearnXors  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainXors {

    public static void main(String[] args) {

        val net = new H1Net(new SigmoidMath(1.8, 0.8));
        val weights = TrainCommon.weights();
        final Function<double[], Double> fun = i -> xor(i[0], i[1]);
        TrainCommon.train(net, BPOINTS, trainSet(fun), weights, 10_000, fun);
    }
}