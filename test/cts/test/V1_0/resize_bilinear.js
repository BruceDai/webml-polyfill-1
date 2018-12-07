describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('check result for Resize bilinear example/1', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op1_value = [1.0, 1.0, 2.0, 2.0];
    const op2_expect = [1.0, 1.0, 1.0, 1.666666667, 1.666666667, 1.666666667, 2.0, 2.0, 2.0];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
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

  it('check result for Resize bilinear example/2', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op1_value = [1.0, 1.0, 2.0, 2.0];
    const op2_expect = [1.0, 1.0, 2.0, 2.0];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
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

  it('check result for Resize bilinear example/3', async function() {
    const model = await nn.createModel(options);
    let operandIndex = 0;
    const op1_value = [1, 3, 5, 7, 9, 7, 5, 3, 1];
    const op2_expect = [1, 4, 6, 5];
    const type2 = {type: nn.INT32};
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    const type0_length = product(type0.dimensions);
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
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
