package ml.lectures.helloworld;

import lombok.val;
import ml.lectures.helloworld.api.H1Net;
import ml.lectures.helloworld.api.SigmoidMath;

import java.util.function.Function;

import static ml.lectures.helloworld.TrainCommon.BPOINTS;
import static ml.lectures.helloworld.api.Utils.nand;

/**
 * TrainNand  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class TrainNand {

    public static void main(String[] args) {

//        val net = new H1Net(new SigmoidMath(1.8, 0.8));
        val net = new H1Net(new SigmoidMath(1.7, 0.8));
        val weights = TrainCommon.weights2();
        final Function<double[], Double> fun = i -> nand(i[0], i[1]);
        TrainCommon.train(net, BPOINTS, TrainCommon.trainSet(fun), weights, 10_000, fun);
    }

}