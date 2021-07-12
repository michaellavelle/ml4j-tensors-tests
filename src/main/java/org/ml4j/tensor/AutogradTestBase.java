package org.ml4j.tensor;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.BackwardConfig;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public abstract class AutogradTestBase<T extends Tensor<T, ?>, V extends Tensor<V, D>, D extends TensorOperations<D>> extends TestBase<V, D> {

    protected abstract boolean isNativeGradientSupported();

    protected abstract boolean isNativeGradientExpected();

    protected AutogradValueRegistry registry;

    @Before
    public void setUp() {
        this.registry = AutogradValueRegistry.create(AutogradTestBase.class.getName());
    }

    @Override
    protected abstract D createData(float value);

    @Override
    protected abstract D createData(float value, Size size);

    protected abstract V createRandomValue(boolean requires_grad, int... dims);
    protected abstract V createOnesValue(boolean requires_grad, int... dims);

    protected T createRandomTensor(boolean requires_grad, int... dims) {
        return createWrappedTensor(createRandomValue(requires_grad, dims));
    }
    protected T createOnesTensor(boolean requires_grad, int... dims) {
        return createWrappedTensor(createOnesValue(requires_grad, dims));
    }

    protected abstract T createWrappedTensor(V tensor);

    @Test
    public void test_scalartensor_addition() {
        var a = createRandomTensor(true, 2, 2);

        //var a = torch.randn(2, 2).requires_grad_(true);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createGradValue(1, false, new Size(2, 2)).mul(2f).getDataAsFloatArray(), 0.0001f);



    }

    @Test
    public void test_scalartensor_addition_second_without_requires_grad() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(false);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));


        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

    }


    @Test
    public void test_scalartensor_addition_first_without_requires_grad() {
        var a = createRandomTensor(false, 2, 2);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(c.requires_grad());
        Assert.assertFalse(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        Assert.assertNull(a.grad());


    }


    @Test
    public void test_scalartensor_addition_reversed() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);



    }


    @Test
    public void test_both_scalartensor_addition() {
        var a = createRandomTensor(true).name_("a");
        var b = createRandomTensor(true).name_("b");

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b).name_("c");

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false).mul(2f).getDataAsFloatArray(), 0.0001f);
    }

    @Test
    public void test_both_scalartensor_addition_second_without_requires_grad() {
        var a = createRandomTensor(true);
        var b = createRandomTensor(false);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        c.backward(createOnesTensor(false).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false).mul(2f).getDataAsFloatArray(), 0.0001f);


    }

    protected void assertArrayEqual(float[] actual, float[] expected, float delta) {
        Assert.assertArrayEquals(expected, actual, delta);
    }

    @Test
    public void test_both_scalartensor_addition_first_without_requires_grad() {
        var a = createRandomTensor(false);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(c.requires_grad());
        Assert.assertFalse(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        Assert.assertNull(a.grad());
    }


    @Test
    public void test_both_scalartensor_addition_reversed() {
        var a = createRandomTensor(true);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false).mul(2f).getDataAsFloatArray(), 0.0001f);


    }

    @Test
    public void test_scalarbroadcast_addition() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        
        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 0);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);


    }



    @Test
    public void test_scalarbroadcast_addition_second_without_requires_grad() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(false, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);


    }


    @Test
    public void test_scalarbroadcast_addition_first_without_requires_grad() {
        var a = createRandomTensor(false, 2, 2);
        var b = createRandomTensor(true, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(c.requires_grad());
        Assert.assertFalse(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 2);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        Assert.assertNull(a.grad());

    }


    @Test
    public void test_scalarbroadcast_addition_reversed() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(true, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);
        System.out.println("Result size:" + c.size());

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        Assert.assertTrue(b.grad().size().dimensions().length == 2);
        Assert.assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

    }


    @Test
    public void test_tensor_addition() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));


        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

    }

    @Test
    public void test_tensor_broadcast_addition() {
        var a = createRandomTensor(true, 2, 128, 128);
        var b = createRandomTensor(true, 1, 128, 128);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 128, 128).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 128, 128).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 1, 128, 128).mul(4f).getDataAsFloatArray(), 0.0001f);


    }

    @Test
    public void test_tensor_broadcast_addition2() {
        var a = createRandomTensor(true, 2, 128, 65);
        var b = createRandomTensor(true, 1, 65);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 128, 65).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 128, 65).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 1, 65).mul(512f).getDataAsFloatArray(), 0.0001f);

    }

    @Test
    @Ignore
    public void test_tensor_filter() {
        var a = createOnesTensor(true, 2, 3);
        var b = a.getTensor(new int[] {0, 1}, new int[] {1, 3});

        Assert.assertEquals(2, b.size().dimensions().length);
        Assert.assertEquals(1, b.size().dimensions()[0]);
        Assert.assertEquals(2, b.size().dimensions()[1]);


        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        b.backward(createOnesTensor(false, 1, 2));

        assertArrayEqual(a.grad().getDataAsFloatArray(), new float[] {0, 1, 1, 0, 0, 0}, 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }

    @Test
    public void test_tensor_addition_second_without_requires_grad() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(false, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));


        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        Assert.assertTrue(a.requires_grad());
        Assert.assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

   
    }

    @Test
    public void test_tensor_addition_first_without_requires_grad() {
        var a = createRandomTensor(false, 2, 2);
        var b = createRandomTensor(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        var c = a.add(b);

        Assert.assertTrue(c.requires_grad());
        Assert.assertFalse(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        Assert.assertNull(a.grad());


    }

    @Test
    public void test_tensor_addition_reversed() {
        var a = createRandomTensor(true, 2, 2);
        var b = createRandomTensor(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        Assert.assertTrue(a.requires_grad());
        Assert.assertTrue(b.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad(false).isNativeGradient());
        }

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

    }


    @Test
    public void test_scalar_addition() {
        var a = createRandomTensor(true, 2, 2);
        var b = (float) Math.random();

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertTrue(a.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad(false).isNativeGradient());
        }

        Assert.assertTrue(a.requires_grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesTensor(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

    }

    @Test(expected = IllegalStateException.class)
    public void test_scalar_addition_without_requires_grad() {
        var a = createRandomTensor(false, 2, 2);
        var b = (float) Math.random();

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        Assert.assertFalse(a.requires_grad());
        Assert.assertFalse(c.requires_grad());

        c.backward(createOnesTensor(false, 2, 2).mul(2f));
       
        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
   
    }


    @Test
    public void test_requires_grad_inplace() {
        var a = createRandomTensor(false, 5, 5);
        var b = createRandomTensor(true, 5, 5);

        a = a.add(b);

        Assert.assertTrue(a.requires_grad());

        // non-leaf
        a = createRandomTensor(false, 5, 5).add(0f);
        b = createRandomTensor(true, 5, 5);
        a = a.add(b);
        Assert.assertTrue(a.requires_grad());

    }

    @Test
    public void test_hessian_vector() {

        var x = createRandomTensor(true, 2, 2);
        var y = createRandomTensor(true, 2, 2);

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));
        z.backward(createOnesTensor(false, 2, 2), new BackwardConfig().with_keep_graph(true)); // create_graph=True

        //with torch.no_grad():
        x.requires_grad_(false);
        y.requires_grad_(false);

        var x_grad = x.mul(2).add(y);
        var y_grad = x.add(y.mul(2));

        assertArrayEqual(x.grad(false).getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad(false).getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);

        x.requires_grad_(true);
        y.requires_grad_(true);

        var grad_sum = x.grad().mul(2).add(y.grad());

        grad_sum.backward(createOnesTensor(false, 2, 2));
        var x_hv = createOnesTensor(false, 2, 2).mul(5); // Should be ones not zeros with create graph
        var y_hv = createOnesTensor(false, 2, 2).mul(4); // Should be ones not zeros with create graph

        assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.add(x_hv).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.add(y_hv).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }
    }

    @Test(expected = IllegalStateException.class)
    public void test_hessian_vector_without_create_graph() {

        var x = createRandomTensor(true, 2, 2);
        var y = createRandomTensor(true, 2, 2);

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));

        z.backward(createOnesTensor(false, 2, 2)); // create_graph=False

        //with torch.no_grad():
        x.requires_grad_(false);
        y.requires_grad_(false);

        var x_grad = x.mul(2).add(y);
        var y_grad = x.add(y.mul(2));

        assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);

        x.requires_grad_(true);
        y.requires_grad_(true);

        var grad_sum = x.grad().mul(2).add(y.grad());

        grad_sum.backward(createOnesTensor(false, 2, 2));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }
    }
}
