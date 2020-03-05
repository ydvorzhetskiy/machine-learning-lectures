package ml.lectures.helloworld;

import lombok.val;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * OneLayerMachineTest  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class OneLayerMachineTest {

    @Test
    public void testTrain() {
        val sut = new OneLayerMachine();
        final int[] weights = {
            //i0 - h0
            1,
            //i0 - h1
            2,
            //i1 - h0
            3,
            //i1 - h1
            4,
            //b0 - h0
            8,
            //b0 - h1
            7,
            //h0 - o0
            5,
            //h1 - h0
            6};
        sut.setHsize(2)
            .setIws(new int[] {0, 2})
            .setBws(new int[] {4, 1})
            .setOws(new int[] {6, 1});

        assertEquals(weights[sut.ihw(0, 0)], 1);
        assertEquals(weights[sut.ihw(0, 1)], 2);
        assertEquals(weights[sut.ihw(1, 0)], 3);
        assertEquals(weights[sut.ihw(1, 1)], 4);

        assertEquals(weights[sut.bhw(0, 0)], 8);
        assertEquals(weights[sut.bhw(0, 1)], 7);

        assertEquals(weights[sut.how(0, 0)], 5);
        assertEquals(weights[sut.how(1, 0)], 6);

        sut.setWeights(new double[] {0.5, 0.3, -0.5, 0.5, 0.1, 0.1, 0.2, 0.3})
            .setMath(new SigmoidMath(0.7, 0.3));

        val set = new double [][] {
            {0, 0},
            {0, 1},
            {1, 1},
            {1, 0}
        };

        val ideal = new double [] {
            0 ^ 0,
            0 ^ 1,
            1 ^ 1,
            1 ^ 0
        };

        sut.train(set, ideal);
        for(double w: sut.getWeights()) {
            System.out.println(w);
        }
    }
}