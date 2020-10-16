// Converted test case (from: conv_float.mod.py). Do not edit
describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float) test', async function() {
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op2 = builder.constant({type: 'float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = builder.add(intermediateOutput1, op3);
    const model = await builder.createModel({op4});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    checkOutput(outputs.op4.buffer, expected);
  });
});
