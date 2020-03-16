package ml.lectures.helloworld.api;

import lombok.val;

import java.util.function.DoubleBinaryOperator;

import static java.lang.System.arraycopy;
import static java.lang.System.out;
import static java.util.Arrays.copyOf;
import static java.util.Arrays.fill;


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

    public static double[] array(final double... d) {
        val res = new double[d.length];
        arraycopy(d, 0, res, 0, d.length);
        return res;
    }
    public static double random() {
        return Math.random();
    };

    public static double[][] sum(final double[][] a, final double[][] b) {
        val res = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            res[i] = new double[a[i].length];
            for (int j = 0; j < a[i].length; j++) {
                res[i][j] = a[i][j] + b[i][j];
            }
        }
        return res;
    }

    public static double sum(final double[] arr) {
        double res = 0.;
        for (final double v : arr) {
            res += v;
        }
        return res;
    }

    public static double[][] vec2matrix(final double[] v) {
        val res = new double[1][];
        res[0] = copyOf(v, v.length);
        return res;
    }

    public static double[][] transpon(final double[] v) {
        val res = new double[v.length][1];
        for (int i = 0; i < v.length; i++) {
            res[i][0] = v[i];
        }
        return res;
    }

    public static double[][] mult(final double[][] x, final double[][] y) {

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

    public static double[] mult(final double[] x, final double[][] y) {

        return mult(vec2matrix(x), y)[0];
    }

    public static double[] add(final double[] x, final double[] y) {

        val res = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            res[i] = x[i] + y[i];
        }
        return res;
    }

    public static double[] evector(int size) {
        val res = new double[size];
        fill(res, 1.);
        return res;
    }

    public static double[][] operate(final double[][] a,
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

    public static double[] operate(final double[] a,
                                   final double[] b,
                                   final DoubleBinaryOperator operator) {

        val x = a.length;
        val res = new double[x];
        for (int i = 0; i < x; i++) {
            res[i] = operator.applyAsDouble(a[i], b[i]);
        }
        return res;
    }

    public static double[][] join(final double[] a,
                                  final double[] b,
                                  final DoubleBinaryOperator operator) {

        val res = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                res[i][j] = operator.applyAsDouble(a[i], b[j]);
            }
//                deltas.h2o(j, i,
//                    math.dweight(
//                        math.gradient(hl.out(j), odeltas[i]),
//                        deltas.h2o(j, i)
//                    )
//                );
//            }
        }
        return res;
    }

    public static double[][] columnv(final double[] a) {

        val res = new double[a.length][1];
        for (int i = 0; i < a.length; i++) {
            res[i][0] = a[i];
        }
        return res;
    }

    public static double[] rowv(final double[][] a, int index) {

        val res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i][index];
        }
        return res;
    }
}