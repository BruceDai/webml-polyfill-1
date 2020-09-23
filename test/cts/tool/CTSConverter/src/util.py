#!/usr/bin/python3

from enum import Enum
from enum import IntEnum
import os

import test_generator as tg

class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

class OperandTypeMapping(NoValue):
    INT32 = 'int32'
    UINT32 = 'uint32'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    TENSOR_INT32 = 'tensor-int32'
    TENSOR_FLOAT16 = 'tensor-float16'
    TENSOR_FLOAT32 = 'tensor-float32'
    TENSOR_OEM_BYTE = 'tensor-quant8-asymm'
    TENSOR_QUANT8_ASYMM = 'tensor-quant8-asymm'


class OperationsMapping(NoValue):
    ADD = 'add'
    AVERAGE_POOL_2D = 'averagePool2d'
    CONCATENATION = 'concat'
    CONV_2D = 'conv2d'
    DEPTHWISE_CONV_2D = 'conv2d'
    # DEPTH_TO_SPACE = 
    # DEQUANTIZE = 
    # EMBEDDING_LOOKUP = 
    # FLOOR = 
    # FULLY_CONNECTED = 
    # HASHTABLE_LOOKUP = 
    # L2_NORMALIZATION = 
    # L2_POOL_2D = 
    # LOCAL_RESPONSE_NORMALIZATION = 
    # LOGISTIC = 
    # LSH_PROJECTION = 
    # LSTM = 
    MAX_POOL_2D = 'maxPool2d'
    MUL = 'mul'
    RELU = 'relu'
    # RELU1 = 
    # RELU6 = 
    RESHAPE = 'reshape'
    # RESIZE_BILINEAR = 
    # RNN = 
    SOFTMAX = 'softmax'
    # SPACE_TO_DEPTH = 
    # SVDF = 
    # TANH = 
    # BATCH_TO_SPACE_ND = 
    TRANSPOSE = 'transpose'
    # ARGMAX = 
    # MAXIMUM = 
    # PRELU =  


class OperationsParmLen(IntEnum):
    ADD = 1 # activation
    CONV_2D_EXPLICIT = 7 # left padding, right padding, top padding, bottom padding, stride width, stride height, activation
    CONV_2D_IMPLICIT = 4 # padding scheme, stride width, stride height, activation


class PaddingCode(IntEnum):
    SAME = 1
    VALID = 2


class ReluType(IntEnum):
    NONE = 0
    RELU = 1
    RELU1 = 2
    RELU6 = 3


class TypedArrayTypeMapping(NoValue):
    FLOAT32 = 'Float32Array'
    INT32 = 'Int32Array'
    TENSOR_FLOAT32 = 'Float32Array'
    TENSOR_INT32 = 'Int32Array'
    TENSOR_QUANT8_ASYMM = 'Uint8Array'
    TENSOR_QUANT8_SYMM_PER_CHANNEL = 'Int8Array'
    TENSOR_QUANT8_ASYMM_SIGNED = 'Int8Array'


def WriteLineToFile(content, fileName):
    print(content, file = fileName)


def WriteMochaTestCaseHeads(fileName):
    comment = "// Converted test case (from: {spec_file}). Do not edit"
    specFileBase = os.path.basename(tg.FileNames.specFile)
    WriteLineToFile(comment.format(spec_file = specFileBase), fileName)
    WriteLineToFile("describe('CTS-v2', function() {", fileName)
    WriteLineToFile("  const assert = chai.assert;", fileName)
    WriteLineToFile("  const nn = navigator.ml.getNeuralNetworkContext('v2');", fileName)


def CheckReluSupported(reluType):
    if reluType == ReluType.RELU:
        return True
    elif reluType == ReluType.RELU1:
        return False # current Web Neural Network API doesn't support relu1
    elif reluType == ReluType.RELU6:
        return False # current Web Neural Network API doesn't support relu6


def GetOperandDesc(t, filter = False):
    dimensions = t.GetDimensionsString()[1:-1].split(',')

    if filter:
        # change [depth_out, filter_height, filter_width, depth_in]
        # to required "nhwc": filter tensor: [height, width, input_channels/groups, output_channels]
        depthOut = dimensions[0]
        dimensions = dimensions[1:4]
        dimensions.append(depthOut)    

    if t.scale == 0.0 and t.zeroPoint == 0 and t.extraParams is None:
        if t.type in ["FLOAT32", "INT32", "UINT32"]:
            return "{type: '%s'}" % OperandTypeMapping[t.type].value
        else:

            return "{type: '%s', dimensions: [%s]}" % (
                OperandTypeMapping[t.type].value,
                ', '.join(dimensions))
    else:
        if t.extraParams is None or t.extraParams.hide:
            return "{type: '%s', dimensions: [%s], scale: %s, zeroPoint: %d}" % (
                OperandTypeMapping[t.type].value, 
                ', '.join(dimensions), 
                t.scale, 
                t.zeroPoint)
        else:
            return "{type: '%s', dimensions: [%s]}" % (
                OperandTypeMapping[t.type].value, 
                ', '.join(dimensions))


def WriteConstantOprandLine(op, fileName, filter = False):
    WriteLineToFile("    const %s = nn.constant(%s, new %s(%s));" % (
        op,
        GetOperandDesc(op.type, filter),
        TypedArrayTypeMapping[op.type.type].value,
        op.value), fileName) # op.value