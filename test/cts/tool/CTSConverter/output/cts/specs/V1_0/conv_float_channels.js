// Converted test case (from: conv_float_channels.mod.py). Do not edit
describe('CTS-v2', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('conv2d (float channels) + add + relu example', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 1, 1, 3]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const act = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const intermediateOutput2 = nn.add(intermediateOutput1, op3);
    const op4 = nn.relu(intermediateOutput2);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([99.0, 99.0, 99.0]));
    const expected = [297.0, 594.0, 891.0];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});
