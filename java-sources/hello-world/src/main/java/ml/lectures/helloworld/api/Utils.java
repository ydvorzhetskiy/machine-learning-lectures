package ml.lectures.helloworld.api;

import lombok.val;

import static java.lang.System.out;


/**
 * Utils  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public final class Utils {

    public static void dump(final Weights weights) {

        val buf = new StringBuilder();
        for (int i = 0; i < weights.isize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\t%.3f", weights.i2h(i, j)));
            }
        }

        for (int i = 0; i < weights.bsize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\t%.3f", weights.b2h(i, j)));
            }
        }

        for (int i = 0; i < weights.osize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\t%.3f", weights.h2o(j, i)));
            }
        }

        out.println(buf.toString());
    }

    public static void dumpLegend(final Weights weights) {

        val buf = new StringBuilder();
        for (int i = 0; i < weights.isize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\ti%d-h%d", i, j));
            }
        }

        for (int i = 0; i < weights.bsize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\tb%d-h%d", i, j));
            }
        }

        for (int i = 0; i < weights.osize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                buf.append(String.format("\th%d-o%d", j, i));
            }
        }

        out.println(buf.toString());
    }

    public static double xor(final double i, final double j) {

        int k = (int ) i;
        int l = (int ) j;
        return k ^ l;
    }

    public static double or(final double i, final double j) {

        int k = (int ) i;
        int l = (int ) j;
        return k | l;
    }

    public static double nand(final double i, final double j) {

        return and(i, j) == 0 ? 1 : 0;
    }

    public static double and(final double i, final double j) {

        int k = (int ) i;
        int l = (int ) j;
        return (k & l);
    }

    public static void randomizeWeights(final Weights weights) {

        for (int i = 0; i < weights.isize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                weights.i2h(i, j, random());
            }
        }

        for (int i = 0; i < weights.bsize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                weights.b2h(i, j, random());
            }
        }

        for (int i = 0; i < weights.osize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                weights.h2o(j, i, random());
            }
        }
    }

    static double random() {
        return Math.random();
    };
}