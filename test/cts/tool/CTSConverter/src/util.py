#!/usr/bin/python3

from enum import Enum
from enum import IntEnum
import os
import numpy as np

# import test_generator as tg

class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

class OperandTypeMapping(NoValue):
    INT32 = 'int32'
    UINT32 = 'uint32'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    TENSOR_INT32 = 'int32'
    TENSOR_FLOAT16 = 'float16'
    TENSOR_FLOAT32 = 'float32'
    TENSOR_OEM_BYTE = 'uint8'
    TENSOR_QUANT8_ASYMM = 'uint8'


OperationsInfoDict = {
    'ADD': {
        'equalOp': 'add',
        'paramList': ['input', 'input', 'activation']
    },
    'CONV_2D': {
        'equalOp': 'conv2d',
        # ['input', 'filter', 'bias', 'padding left', 'padding right', 'padding top', 'padding bottom', 'stride width', 'stride height', 'activation']
        'paramList': ['input', 'filter', 'bias', 'padding', 'padding', 'padding', 'padding', 'stride', 'stride', 'activation'] # current only for explicit paddings
    },
    'DEPTHWISE_CONV_2D': {
        'equalOp': 'conv2d',
        # ['input', 'filter', 'bias', 'padding left', 'padding right', 'padding top', 'padding bottom', 'stride width', 'stride height', 'multiplier', 'activation']
        'paramList': ['input', 'filter', 'bias', 'padding', 'padding', 'padding', 'padding', 'stride', 'stride', 'multiplier', 'activation'] # current only for explicit paddings
    }
}


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
    # print(content)
    print(content, file = fileName)


def WriteMochaTestCaseHeads(fileName):
    WriteLineToFile("'use strict';", fileName)
    WriteLineToFile("import * as utils from '../utils.js';\n", fileName)
    WriteLineToFile("describe('CTS', function() {", fileName)
    WriteLineToFile("  const nn = navigator.ml.getNeuralNetworkContext();",
                    fileName)
    WriteLineToFile("  const builder = nn.createModelBuilder();", fileName)


def CheckFilterOp(androidNNOpType, index):
    return OperationsInfoDict[androidNNOpType]['paramList'][index] == 'filter'


def ConvertDimensions(dimensions, groups = 1, layout = 'nhwc'):
    if layout == 'nhwc':
        # Convert dimensions of "nhwc" [depth_out, filter_height, filter_width, depth_in]
        # to required:
        # [height, width, input_channels/groups, output_channels]
        depthOut = dimensions[0]
        depthIn = dimensions[-1]
        dimensions = dimensions[1:3]
        dimensions.append(str(int(int(depthIn) / groups)))
        dimensions.append(depthOut)

    return dimensions

# TODO Need update for coming scalar tensor define
def GetOperandDesc(t, filter = False, layout = 'nhwc'):
    dimensions = t.GetDimensionsString()[1:-1].split(',')
    dimensions = [d.strip() for d in dimensions]

    if filter and layout == 'nhwc':
        dimensions = ConvertDimensions(dimensions)

    if t.scale == 0.0 and t.zeroPoint == 0 and t.extraParams is None:
        if t.type in ["FLOAT32", "INT32", "UINT32"]:
            return "{type: '%s'}" % OperandTypeMapping[t.type].value
        else:

            return "{type: '%s', dimensions: [%s]}" % (
                OperandTypeMapping[t.type].value,
                ', '.join(dimensions))
    else:
        if t.extraParams is None or t.extraParams.hide:
            # TODO handle scalar tensor
            return "{type: '%s', dimensions: [%s], scale: %s, zeroPoint: %d}" \
             % (OperandTypeMapping[t.type].value,
                ', '.join(dimensions),
                t.scale,
                t.zeroPoint)
        else:
            return "{type: '%s', dimensions: [%s]}" % \
             (OperandTypeMapping[t.type].value,
              ', '.join(dimensions))


def reorderValues(values, t):
    dimensions = t.GetDimensionsString()[1:-1].split(',')
    dimensions = ConvertDimensions(dimensions)
    dimensions = [int(i) for i in dimensions]
    valuesArray = np.array(values)
    return list(np.reshape(valuesArray, tuple(dimensions), order = 'F').ravel())
