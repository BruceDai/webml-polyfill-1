// Converted test case (from: add.mod.py). Do not edit
describe('CTS-v2', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('add example', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [2]});
    const op2 = nn.input('op2', {type: 'tensor-float32', dimensions: [2]});
    const op3 = nn.add(op1, op2);
    const model = await nn.createModel([{name: 'op3', operand: op3}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 2.0]));
    execution.setInput('op2', new Float32Array([3.0, 4.0]));
    const expected = [4.0, 6.0];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op3', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});
