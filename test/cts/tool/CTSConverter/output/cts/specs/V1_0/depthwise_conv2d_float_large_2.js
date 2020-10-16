'use strict';
import * as utils from '../utils.js';

describe('CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  it('conv2d + add test converted from Android NNAPI CTS-DEPTHWISE_CONV_2D (float large) test/2', async function() {
    // Converted test case (from: depthwise_conv2d_float_large_2.mod.py). Do not edit
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op2 = builder.constant({type: 'float32', dimensions: [2, 2, 4, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25, 10.0, 20.0, 30.0, 40.0, 0.0, 1.0, 0.0, 1.0, 100.0, 100.0, 100.0, 100.0]));
    const op3 = builder.constant({type: 'float32', dimensions: [4]}, new Float32Array([600000, 700000, 800000, 900000]));
    const pad0 = 0;
    const stride = 1;
    const channelMultiplier = 1;
    const expected = [600010, 700046, 830000, 900000];
    const intermediateOutput1 = builder.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], channelMultiplier, 'nhwc');
    const op4 = builder.add(intermediateOutput1, op3);
    const model = await builder.createModel({op4});
    const compilation = await model.compile({powerPreference: 'low-power'});
    const op1Buffer = new Float32Array([10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0]);
    let outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected);
  });
});
