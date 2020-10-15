describe('converted CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 3, 3, 1]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [1]}, new Float32Array([0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]));
    const expected = [0.875, 0.875, 0.875, 0.875];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float channels) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 1, 1, 3]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
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

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float channels weights as inputs) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 1, 1, 3]});
    const op2 = nn.input('op2', {type: 'tensor-float32', dimensions: [1, 1, 3, 3]});
    const op3 = nn.input('op3', {type: 'tensor-float32', dimensions: [3]});
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([99.0, 99.0, 99.0]));
    execution.setInput('op2', new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    execution.setInput('op3', new Float32Array([0.0, 0.0, 0.0]));
    const expected = [297.0, 594.0, 891.0];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float large) test', async function() {
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

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float large weights as inputs) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 2, 3, 3]});
    const op2 = nn.input('op2', {type: 'tensor-float32', dimensions: [1, 1, 3, 3]});
    const op3 = nn.input('op3', {type: 'tensor-float32', dimensions: [3]});
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]));
    execution.setInput('op2', new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    execution.setInput('op3', new Float32Array([0.0, 0.0, 0.0]));
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });

  it('conv2d + add test converted from Android NNAPI CTS-CONV_2D (float weights as inputs) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 3, 3, 1]});
    const op2 = nn.input('op2', {type: 'tensor-float32', dimensions: [2, 2, 1, 1]});
    const op3 = nn.input('op3', {type: 'tensor-float32', dimensions: [1]});
    const pad0 = 0;
    const stride = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], 1, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]));
    execution.setInput('op2', new Float32Array([0.25, 0.25, 0.25, 0.25]));
    execution.setInput('op3', new Float32Array([0]));
    const expected = [0.875, 0.875, 0.875, 0.875];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });

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

  it('conv2d + add test converted from Android NNAPI CTS-DEPTHWISE_CONV_2D (float large) test/2', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 2, 2, 4]});
    const op2 = nn.constant({type: 'tensor-float32', dimensions: [2, 2, 4, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25, 10.0, 20.0, 30.0, 40.0, 0.0, 1.0, 0.0, 1.0, 100.0, 100.0, 100.0, 100.0]));
    const op3 = nn.constant({type: 'tensor-float32', dimensions: [4]}, new Float32Array([600000, 700000, 800000, 900000]));
    const pad0 = 0;
    const stride = 1;
    const channelMultiplier = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], channelMultiplier, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0]));
    const expected = [600010, 700046, 830000, 900000];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });

  it('conv2d + add test converted from Android NNAPI CTS-DEPTHWISE_CONV_2D (float large 2 weights as inputs) test', async function() {
    const op1 = nn.input('op1', {type: 'tensor-float32', dimensions: [1, 2, 2, 4]});
    const op2 = nn.input('op2', {type: 'tensor-float32', dimensions: [2, 2, 4, 1]});
    const op3 = nn.input('op3', {type: 'tensor-float32', dimensions: [4]});
    const pad0 = 0;
    const stride = 1;
    const channelMultiplier = 1;
    const intermediateOutput1 = nn.conv2d(op1, op2, [pad0, pad0, pad0, pad0], [stride, stride], [1, 1], channelMultiplier, 'nhwc');
    const op4 = nn.add(intermediateOutput1, op3);
    const model = await nn.createModel([{name: 'op4', operand: op4}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('op1', new Float32Array([10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0]));
    execution.setInput('op2', new Float32Array([0.25, 0.25, 0.25, 0.25, 10.0, 20.0, 30.0, 40.0, 0.0, 1.0, 0.0, 1.0, 100.0, 100.0, 100.0, 100.0]));
    execution.setInput('op3', new Float32Array([600000, 700000, 800000, 900000]));
    const expected = [600010, 700046, 830000, 900000];
    const outputBuffer = new Float32Array(expected.length);
    execution.setOutput('op4', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  });
});