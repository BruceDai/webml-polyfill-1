describe('Abnormal Test', function() {
  const assert = chai.assert;
  let nn, operandIndex;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
    operandIndex = 0;
  });

  describe('#addOperation API', function() {
    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);

        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);

        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.ADD, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);

        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);

        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.ADD, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "AVERAGE_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "AVERAGE_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "AVERAGE_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "AVERAGE_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, padingcode, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, padingcode, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);

        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MAX_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MAX_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MAX_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MAX_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);

        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);

        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);

        model.setOperandValue(fusecode, new Int32Array([4]));

        assert.throws(() => {
          model.addOperation(nn.MUL, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};

        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);

        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);

        model.setOperandValue(fusecode, new Int32Array([-1]));

        assert.throws(() => {
          model.addOperation(nn.MUL, [op0, op1, fusecode], [op3]);
        });
      });
    });
  });
});
