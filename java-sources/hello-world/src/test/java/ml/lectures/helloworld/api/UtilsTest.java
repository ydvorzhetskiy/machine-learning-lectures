package ml.lectures.helloworld.api;

import lombok.val;
import org.testng.annotations.Test;

import static ml.lectures.helloworld.api.Utils.mult;
import static org.testng.Assert.*;

/**
 * UtilsTest
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public class UtilsTest {

    @Test
    public void testMult() {
        val arr = mult(
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
    }
}