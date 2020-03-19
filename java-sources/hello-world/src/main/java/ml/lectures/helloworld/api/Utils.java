package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.DoubleBinaryOperator;

import static java.lang.System.arraycopy;
import static java.lang.System.out;
import static java.util.Arrays.fill;


/**
 * Utils  description
 *
 * @author <a href="mailto:oslautin@luxoft.com">Oleg N.Slautin</a>
 */
public final class Utils {

    public static final DoubleBinaryOperator SUMOP = Double::sum;

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

    public static double random() {
        return Math.random();
    }

    public static double sum(final double[] arr) {
        double res = 0.;
        for (final double v : arr) {
            res += v;
        }
        return res;
    }

    public static double[][] tran(final double[][] v) {
        val w = v[0].length;
        val h = v.length;
        val res = new double[w][h];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                res[j][i] = v[i][j];
            }
        }
        return res;
    }

    public static double[][] mul(final double[][] x, final double[][] y) {

        val xl = x.length;
        val yl = y.length;
        val yw = y[0].length;
        val res = new double[xl][yw];
        for (int i = 0; i < xl; ++i) {
            fill(res[i], 0.);
            for (int k = 0; k < yl; ++k) {
                double[] b = y[k];
                double a = x[i][k];
                for (int j = 0; j < yw; ++j) {
                    res[i][j] += a * b[j];
                }
            }
        }
        return res;
    }

    public static double[][] sum(final double[][] a,
                                 final double[][] b) {

        return oper(a, b, SUMOP);
    }

    public static double[] ecol(int size) {
        val res = new double[size];
        fill(res, 1.);
        return res;
    }

    public static double[][] oper(final double[][] a,
                                  final double[][] b,
                                  final DoubleBinaryOperator operator) {

        val x = a.length;
        val y = a[0].length;
        val res = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                res[i][j] = operator.applyAsDouble(a[i][j], b[i][j]);
            }
        }
        return res;
    }

    public static double[][] col(final double[] a) {

        val res = new double[a.length][1];
        for (int i = 0; i < a.length; i++) {
            res[i][0] = a[i];
        }
        return res;
    }

    public static double[][] row(final double[] a) {

        val res = new double[1][a.length];
        arraycopy(a, 0, res[0], 0, a.length);
        return res;
    }


    public static double[] row(final double[][] a, int index) {

        return a[index];
    }


    public static double[] col(final double[][] a, int index) {

        val res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i][index];
        }
        return res;
    }
}