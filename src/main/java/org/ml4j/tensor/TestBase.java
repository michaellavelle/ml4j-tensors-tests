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

import org.junit.Before;
import org.jvmpy.symbolictensors.Size;
import org.mockito.MockitoAnnotations;

/**
 * A base test for Tensor implementations.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class TestBase<T extends Tensor<T, D>, D> {

    protected Size size;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        size = new Size(2, 2);
    }

    protected abstract T createGradValue(float value, boolean requires_grad);
   
    protected abstract T createGradValue(D value, boolean requires_grad);

    protected abstract void assertEquals(D value1, D value2);

    protected abstract D add(D value1, D value2);
    protected abstract D mul(D value1, float value2);

    protected abstract D createData(float value);

    protected abstract D createData(float value, Size size);

    protected abstract T createGradValue(float value, boolean requires_grad, Size size);

    protected T one() {
        return createGradValue(1, false);
    }

    protected T ten() {
        return createGradValue(10, false);
    }
}
