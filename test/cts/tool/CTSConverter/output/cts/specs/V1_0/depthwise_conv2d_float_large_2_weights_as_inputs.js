// Converted test case (from: depthwise_conv2d_float_large_2_weights_as_inputs.mod.py). Do not edit
describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('conv2d + add test converted from Android NNAPI CTS-DEPTHWISE_CONV_2D (float large 2 weights as inputs) test', async function() {
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2, 4, 1]});
    const op3 = builder.input('op3', {type: 'float32', dimensions: [4]});
    const pad0 = 0;
    const stride = 1;
    const channelMultiplier = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], channelMultiplier, 'nhwc');
    const op4 = builder.add(intermediateOutput1, op3);
    const model = await builder.createModel({'op4', op4});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0]);
    const op2Buffer = new Float32Array([0.25, 0.25, 0.25, 0.25, 10.0, 20.0, 30.0, 40.0, 0.0, 1.0, 0.0, 1.0, 100.0, 100.0, 100.0, 100.0]);
    const op3Buffer = new Float32Array([600000, 700000, 800000, 900000]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    checkOutput(outputs.op4.buffer, expected);
  });
});
