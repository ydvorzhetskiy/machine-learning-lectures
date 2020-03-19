package ml.lectures.helloworld.api;

import org.testng.annotations.Test;

import static ml.lectures.helloworld.api.Utils.ecol;
import static ml.lectures.helloworld.api.Utils.mul;
import static org.testng.Assert.assertEquals;

/**
 * UtilsTest
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class UtilsTest {

    @Test
    public void testMult() {
        double[][] arr = mul(
            new double[][] {
                {1, 2}, {3, 4}, {5, 6}
            },
            new double[][] {
                {7, 8, 9},
                {10, 11, 12}
            }
        );
        assertEquals(arr[0][0], 27.0);
        assertEquals(arr[2][2], 117.0);

        arr = mul(
            new double[][] {
                ecol(2)
            },
            new double[][] {
                {7, 8, 9},
                {10, 11, 12}
            }
        );

        assertEquals(arr[0][0], 17.0);
        assertEquals(arr[0][2], 21.0);

    }
}