// Converted test case (from: conv_float.mod.py). Do not edit
describe('CTS-v2', function() {
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('conv2d (float) + add example', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 3, 3, 1]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [1]}, new Float32Array([0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]));
    const expected = [0.875, 0.875, 0.875, 0.875];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});
