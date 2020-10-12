// Converted test case (from: conv_float_large.mod.py). Do not edit
describe('CTS-v2', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('conv2d (float large) + add example', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 2, 3, 3]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]));
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});
