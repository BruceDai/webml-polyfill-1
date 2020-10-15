// Converted test case (from: conv_float_large_weights_as_inputs.mod.py). Do not edit
describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float large weights as inputs) test', async function() {
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op3 = builder.input('op3', {type: 'float32', dimensions: [3]});
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = builder.add(intermediateOutput1, op3);
    const model = await builder.createModel({'op4', op4});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    const op2Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    const op3Buffer = new Float32Array([0.0, 0.0, 0.0]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    checkOutput(outputs.op4.buffer, expected);
  });
});
