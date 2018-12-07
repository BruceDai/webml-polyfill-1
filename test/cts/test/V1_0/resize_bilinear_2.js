describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('check result for Resize bilinear example/4', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op1_value = [3, 4, 6, 10, 9, 10, 12, 16];
    const op2_expect = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/5', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [3, 4, 6, 10, 9, 10, 12, 16];
    const op1_value = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/6', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [3, 4, 6, 10, 9, 10, 12, 16];
    const op1_value = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/7', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [1, 3, 5, 7, 9, 7, 5, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16];
    const op1_value = [1, 3, 5, 7, 9, 7, 5, 3, 1, 2, 3, 4, 3.6666667, 5, 6.3333335, 7.6666665, 9, 9, 9, 9, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 4]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 4]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/8', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [1, 3, 5, 7, 9, 7, 5, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16];
    const op1_value = [1, 3, 5, 7, 5, 4.5, 4, 3.5, 3.6666667, 5, 6.3333335, 7.6666665, 9, 9.5, 10, 10.5, 5, 6, 7, 8, 11, 12, 13, 14];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 4]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 4]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/9', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [1, 3, 5, 7, 9, 7, 5, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16];
    const op1_value = [1, 2.3333335, 3, 5, 6.3333335, 7, 9, 7.6666665, 7, 5, 3.6666665, 3, 1, 1.6666667, 2, 3, 3.6666667, 4, 5, 5.6666665, 6, 7, 7.6666665, 8, 9, 9.666667, 10, 11, 11.666667, 12, 13, 13.666667, 14, 15, 15.666667, 16];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 3, 2, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 3, 3, 1]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear example/10', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op2_expect = [1, 3, 5, 7];
    const op1_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 1, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 3, 3, 1]};
    const type1_length = product(type1.dimensions);
    const op1 = operandIndex++;
    model.addOperand(type0);
    const op2 = operandIndex++;
    model.addOperand(type1);
    const height = operandIndex++;
    model.addOperand(type2);
    const width = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);
    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();
    const compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();
    const execution = await compilation.createExecution();
    const op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    const op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
