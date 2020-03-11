package ml.lectures.helloworld.api;

import lombok.val;

/**
 * Utils  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public final class Utils {

    public static void dump(final Weights weights) {
        val out = new StringBuilder();
        for (int i = 0; i < weights.isize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                out.append(String.format("\t%.3f", weights.i2h(i, j)));
            }
        }

        for (int i = 0; i < weights.bsize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                out.append(String.format("\t%.3f", weights.b2h(i, j)));
            }
        }

        for (int i = 0; i < weights.osize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                out.append(String.format("\t%.3f", weights.h2o(j, i)));
            }
        }

        System.out.println(out);
    }

    public static void dumpLegend(final Weights weights) {
        val out = new StringBuilder();
        for (int i = 0; i < weights.isize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
                out.append(String.format("\ti%d-h%d", i, j));
            }
        }

        for (int i = 0; i < weights.hsize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
//                out.append(String.format("\t%.3f", b2h(i, j)));
                out.append(String.format("\tb%d-h%d", i, j));
            }
        }

        for (int i = 0; i < weights.osize(); i++) {
            for (int j = 0; j < weights.hsize(); j++) {
//                out.append(String.format("\t%.3f", h2o(j, i)));
                out.append(String.format("\th%d-o%d", j, i));
            }
        }

        System.out.println(out);
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
}