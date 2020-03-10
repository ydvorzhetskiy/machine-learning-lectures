package ml.lectures.helloworld;

import ml.lectures.helloworld.api.ArrayWeights;
import ml.lectures.helloworld.api.Weights;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;

/**
 * OneLayerMachineTest  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class OneLayerMachineTest {

    @Test
    public void testWeights() {
//        val sut = new OneLayerMachine();
//        final int[] weights = {

//        sut.setHsize(2)
//            .setIws(new int[] {0, 2})
//            .setBws(new int[] {4, 1})
//            .setOws(new int[] {6, 1});

        Weights weights = new ArrayWeights(2, 2, 1);
        //i0 - h0
        weights.i2h(0, 0, 1);
        //i0 - h1
        weights.i2h(0, 1, 2);
        //i1 - h0
        weights.i2h(1, 0, 3);
        //i1 - h1
        weights.i2h(1, 1, 4);
        //b0 - h0
        weights.b2h(0, 0, 5);
        //b0 - h1
        weights.b2h(0, 1, 6);
        //h0 - o0
        weights.h2o(0, 0, 7);
        //h1 - h0
        weights.h2o(1, 0, 8);

        ////
        assertEquals(weights.i2h(0, 0), 1.0);
        assertEquals(weights.i2h(0, 1), 2.0);
        assertEquals(weights.i2h(1, 0), 3.0);
        assertEquals(weights.i2h(1, 1), 4.0);
        assertEquals(weights.b2h(0, 0), 5.0);
        assertEquals(weights.b2h(0, 1), 6.0);
        assertEquals(weights.h2o(0, 0), 7.0);
        assertEquals(weights.h2o(1, 0), 8.0);

//        sut.setWeights(new double[] {0.5, 0.3, -0.5, 0.5, 0.1, 0.1, 0.2, 0.3})
//            .setMath(new SigmoidMath(0.7, 0.3));

//        val set = new double [][] {
//            {0, 0},
//            {0, 1},
//            {1, 1},
//            {1, 0}
//        };
//
//        val ideal = new double [] {
//            0 ^ 0,
//            0 ^ 1,
//            1 ^ 1,
//            1 ^ 0
//        };
//
//        sut.train(set, ideal);
//        for(double w: sut.getWeights()) {
//            System.out.println(w);
//        }
    }
}