// Converted test case (from: conv_float_channels.mod.py). Do not edit
describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('conv2d + add + relu test converted from Android NNAPI CTS-CONV_2D (float channels) test', async function() {
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    const op3 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const intermediateOutput2 = builder.add(intermediateOutput1, op3);
    const op4 = builder.relu(intermediateOutput2);
    const model = await builder.createModel({'op4', op4});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([99.0, 99.0, 99.0]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    checkOutput(outputs.op4.buffer, expected);
  });
});
