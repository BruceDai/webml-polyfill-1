// Converted test case (from: depthwise_conv2d_float_large.mod.py). Do not edit
describe('CTS-v2', function() {
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('conv2d + add test converted from Android NNAPI CTS-DEPTHWISE_CONV_2D (float large) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 2, 2, 2]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [2, 2, 2, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [2]}, new Float32Array([100, 200]));
    const pad0 = 0;
    const stride = 1;
    const channelMultiplier = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], channelMultiplier, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([10, 21, 10, 22, 10, 23, 10, 24]));
    const expected = [110, 246];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});
