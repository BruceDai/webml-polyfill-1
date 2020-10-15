// Converted test case (from: add.mod.py). Do not edit
describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('add test converted from Android NNAPI CTS-ADD test', async function() {
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2]});
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2]});
    const op3 = builder.add(op1, op2);
    const model = await builder.createModel({'op3', op3});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([1.0, 2.0]);
    const op2Buffer = new Float32Array([3.0, 4.0]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}});
    checkOutput(outputs.op3.buffer, expected);
  });
});
