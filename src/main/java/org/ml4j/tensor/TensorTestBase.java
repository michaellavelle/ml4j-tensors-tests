/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.tensor;


import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.MatrixFactory;
import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;

/**
 * A base test for Tensor implementations.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class TensorTestBase<T extends Tensor<T, D>, D> extends TestBase<T, D> {

    protected AutogradValueRegistry registry;

    protected final static MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();

    protected final static DirectedComponentsContext DEFAULT_DIRECTED_COMPONENTS_CONTEXT = new DirectedComponentsContextImpl(DEFAULT_MATRIX_FACTORY, false);

    @Override
    public void setUp() {
        super.setUp();
        this.registry = AutogradValueRegistry.create(TensorTestBase.class.getName());
    }

    protected abstract boolean isNativeGradientSupported();
    protected abstract boolean isNativeGradientExpected();

    protected abstract void assertSize(T tensor, Size s);

    private void assertDataExpectations(Tensor<?, ?> a) {
        float firstRowFirstColumn = a.get(0, 0);
        float firstRowSecondColumn = a.get(0, 1);
        float firstRowThirdColumn = a.get(0, 2);

        float secondRowFirstColumn = a.get(1, 0);
        float secondRowSecondColumn = a.get(1, 1);
        float secondRowThirdColumn = a.get(1, 2);

        Assert.assertEquals(1, firstRowFirstColumn, 0.001f);
        Assert.assertEquals(2, firstRowSecondColumn, 0.001f);
        Assert.assertEquals(3, firstRowThirdColumn, 0.001f);
        Assert.assertEquals(4, secondRowFirstColumn, 0.001f);
        Assert.assertEquals(5, secondRowSecondColumn, 0.001f);
        Assert.assertEquals(6, secondRowThirdColumn, 0.001f);

        Assert.assertEquals(1, a.get(0), 0.001f);
        Assert.assertEquals(2, a.get(1), 0.001f);
        Assert.assertEquals(3, a.get(2), 0.001f);
        Assert.assertEquals(4, a.get(3), 0.001f);
        Assert.assertEquals(5, a.get(4), 0.001f);
        Assert.assertEquals(6, a.get(5), 0.001f);

        var firstRow = a.getTensor(0, -1);

        Assert.assertEquals(1, firstRow.size().dimensions().length);
        Assert.assertEquals(3, firstRow.size().dimensions()[0]);

        Assert.assertEquals(1, firstRow.get(0), 0.001f);
        Assert.assertEquals(2, firstRow.get(1), 0.001f);
        Assert.assertEquals(3, firstRow.get(2), 0.001f);

        var firstColumn = a.getTensor(-1, 0);

        Assert.assertEquals(1, firstColumn.size().dimensions().length);
        Assert.assertEquals(2, firstColumn.size().dimensions()[0]);

        Assert.assertEquals(1, firstColumn.get(0), 0.001f);
        Assert.assertEquals(4, firstColumn.get(1), 0.001f);

        var transposed = a.t();
        float[] transposedData = transposed.getDataAsFloatArray();
        Assert.assertEquals(1, transposedData[0], 0.001f);
        Assert.assertEquals(4, transposedData[1], 0.001f);
        Assert.assertEquals(2, transposedData[2], 0.001f);
        Assert.assertEquals(5, transposedData[3], 0.001f);
        Assert.assertEquals(3, transposedData[4], 0.001f);
        Assert.assertEquals(6, transposedData[5], 0.001f);
    }

    protected abstract T createGradValue(float[] data, int...dims);


    @Test
    @Ignore
    public void test_data() {
        float[] data = new float[] {1, 2, 3, 4, 5, 6};

        var a = createGradValue(data, 2, 3);

        assertDataExpectations(a);

        var sub = a.getTensor(new int[] {0, 0}, new int[]{1, 2});

        Assert.assertEquals(1, sub.size().dimensions().length);
        Assert.assertEquals(2, sub.size().dimensions()[0]);

        var b = a.toDJLTensor();

        assertDataExpectations(b);

        var c = a.toML4JTensor(DEFAULT_DIRECTED_COMPONENTS_CONTEXT);

        assertDataExpectations(c);

        var d = a.toDL4JTensor();

        assertDataExpectations(d);

    }

    @Test
    public void test_reshape() {
        var a = createGradValue(-4f, true, new Size(2, 128)).name_("a");
        var b = a.reshape(new Size(1, 256));

        Assert.assertEquals(a.numel(), b.numel());
        assertEquals(a.data().get(), b.data().get());
        assertSize(a, new Size(2, 128));
        assertSize(b, new Size(1, 256));
    }

    @Test
    public void test_view() {
        var a = createGradValue(-4f, true, new Size(2, 128)).name_("a");
        var b = a.view(new Size(1, 256));

        Assert.assertEquals(a.numel(), b.numel());
        assertEquals(a.data().get(), b.data().get());
        assertSize(a, new Size(2, 128));
        assertSize(b, new Size(1, 256));
    }

    @Test(expected=IllegalArgumentException.class)
    public void test_view_incorrect_size() {
        var a = createGradValue(-4f, true, new Size(2, 128)).name_("a");
        a.view(new Size(2, 256));
    }

    @Test
    public void test_resize_() {
        var a = createGradValue(-4f, true, new Size(2, 128)).name_("a");
        var b = a.resize_(new Size(1, 256));

        Assert.assertSame(a, b);
        assertEquals(a.data().get(), b.data().get());
        assertSize(a, new Size(1, 256));
    }



    @Test
    public void test_example() {

        var a = createGradValue(-4f, true).name_("a");
        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var b = createGradValue(2.0f, true).name_("b");
        if (!isNativeGradientExpected()) {
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        var d = a.mul(b).add(b.mul(b).mul(b));

        c = c.add(c.add(1));

        c = c.add(one().add(c).sub(a));

        d = d.add(d.mul(2).add(b.add(a).relu()));

        d = d.add(d.mul(3).add(b.sub(a).relu()));

        var e = c.sub(d);

        var f = e.mul(e);

        var g = f.div(2f);

        g = g.add(ten().div(f));

        assertEquals(createData(24.70f), g.data().get());

        g.backward();

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
            Assert.assertFalse(c.grad().isNativeGradient());
            Assert.assertFalse(d.grad().isNativeGradient());
            Assert.assertFalse(e.grad().isNativeGradient());
            Assert.assertFalse(f.grad().isNativeGradient());
            Assert.assertFalse(g.grad().isNativeGradient());
        }

        assertEquals(createData(138.83f), a.grad().data().get());

        assertEquals(createData(645.58f), b.grad().data().get());

    }

    @Test
    public void test_hessian_vector2() {

        var x = createGradValue(0.5f, true).name_("x");
        var y = createGradValue(0.5f, true).name_("x");

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.add(y);
        z.backward();

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }
    }

    @Test
    public void test_hessian_vector() {

        var x = createGradValue(0.5f, true).name_("x");

        var y = createGradValue(0.6f, true).name_("y");

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.mul(x).add(y.mul(x).add(y.mul(y))).name_("z");

        var two = createGradValue(2, true).name_("two");

        z.backward(new BackwardConfig().with_keep_graph(true));

        var xGradAfterFirstBackward = x.grad(false);

        var yGradAfterFirstBackward = y.grad(false);

        assertEquals(createData(1.6f), xGradAfterFirstBackward.data().get());

        assertEquals(createData(1.7f), yGradAfterFirstBackward.data().get());

        var x_grad = createGradValue(add(mul(x.data().get(),2f),y.data().get()), false);
        var y_grad = createGradValue(add(x.data().get(), mul(y.data().get(),2f)), false);

        var grad_sum = x.grad().mul(two).add(y.grad());

        grad_sum.backward(new BackwardConfig());

        var xGradAfterSecondBackward = x.grad();

        var yGradAfterSecondBackward = y.grad();

        Assert.assertSame(xGradAfterFirstBackward, xGradAfterSecondBackward);
        assertEquals(createData(6.6f), xGradAfterSecondBackward.data().get());

        Assert.assertSame(yGradAfterFirstBackward, yGradAfterSecondBackward);

        assertEquals(createData(5.7f), yGradAfterSecondBackward.data().get());

        var x_hv = 5;
        var y_hv = 4;

        Assert.assertArrayEquals(x.grad().getDataAsFloatArray(), x_grad.add(createGradValue(x_hv, false)).getDataAsFloatArray(), 0.001f);
        Assert.assertArrayEquals(y.grad().getDataAsFloatArray(), y_grad.add(createGradValue(y_hv, false)).getDataAsFloatArray(), 0.001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }

    }

    @Test
    public void test_sum() {

        var a = createGradValue(-4f, true, new Size(2, 2)).name_("a");

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.sum();

        assertEquals(createData(-16f, new Size()), c.data().get());

        c.backward();

        assertEquals(createData(1, new Size(2, 2)), a.grad(false).data().get());

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }

    @Test
    public void test_get_row() {

        var a = createGradValue(-4f, true, new Size(2, 2)).name_("a");

        var c = a.getTensor(-1, 0);

        assertEquals(createData(-4, new Size(2, 1)), c.data().get());
    }

    @Test
    public void testMatMul() {
        var left = createGradValue(-2, true, new Size(new Size(2, 128), new Size(512))).name_("a");
        var right = createGradValue(1, true, new Size(512, 65)).name_("a");

        if (!isNativeGradientExpected()) {
            left.getGradNode().setDisableNativeGradient(true);
            right.getGradNode().setDisableNativeGradient(true);
        }

        var result = left.matmul(right);
        Assert.assertEquals(3, result.size().dimensions().length);
        Assert.assertEquals(2, result.size().dimensions()[0]);
        Assert.assertEquals(128, result.size().dimensions()[1]);
        Assert.assertEquals(65, result.size().dimensions()[2]);

        result.backward();

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), left.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), right.grad(false).isNativeGradient());
        }

        Assert.assertNotNull(left.grad());
        System.out.println(left.grad().size());
        Assert.assertEquals(3, left.grad().size().dimensions().length);
        Assert.assertEquals(2, left.grad().size().dimensions()[0]);
        Assert.assertEquals(128, left.grad().size().dimensions()[1]);
        Assert.assertEquals(512, left.grad().size().dimensions()[2]);

        Assert.assertNotNull(right.grad(false));
        Assert.assertEquals(2, right.grad(false).size().dimensions().length);
        Assert.assertEquals(512, right.grad(false).size().dimensions()[0]);
        Assert.assertEquals(65, right.grad().size().dimensions()[1]);

    }
}
