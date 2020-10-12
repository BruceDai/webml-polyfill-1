#!/usr/bin/python3

# Copyright 2018, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CTS testcase generator

Implements CTS test backend. Invoked by ml/nn/runtime/test/specs/generate_tests.sh;
See that script for details on how this script is used.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import math
import os
import re
import sys
import traceback

# Stuff from test generator
import test_generator as tg
from test_generator import ActivationConverter
from test_generator import BoolScalar
from test_generator import Configuration
from test_generator import DataTypeConverter
from test_generator import DataLayoutConverter
from test_generator import Example
from test_generator import Float16Scalar
from test_generator import Float32Scalar
from test_generator import Float32Vector
from test_generator import GetJointStr
from test_generator import IgnoredOutput
from test_generator import Input
from test_generator import Int32Scalar
from test_generator import Int32Vector
from test_generator import Internal
from test_generator import Model
from test_generator import Operand
from test_generator import Output
from test_generator import Parameter
from test_generator import ParameterAsInputConverter
from test_generator import RelaxedModeConverter
from test_generator import SmartOpen
from test_generator import SymmPerChannelQuantParams

import util

def IndentedPrint(s, indent=2, *args, **kwargs):
    print('\n'.join([" " * indent + i for i in s.split('\n')]), *args, **kwargs)

# Take a model from command line
def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file/directory")
    parser.add_argument(
        "-m", "--model", help="the output model file/directory", default="-")
    parser.add_argument(
        "-e", "--example", help="the output example file/directory", default="-")
    parser.add_argument(
        "-t", "--test", help="the output test file/directory", default="-")
    parser.add_argument(
        "-c", "--cts", help="the CTS TestGeneratedOneFile.cpp", default="-")
    parser.add_argument(
        "-f", "--force", help="force to regenerate all spec files", action="store_true")
    # for slicing tool
    parser.add_argument(
        "-l", "--log", help="the optional log file", default="")

    # For js
    parser.add_argument(
        "-js", "--jsTest", help="the output javascript file", default="-")
    # end

    args = parser.parse_args()

    ''' Original
    tg.FileNames.InitializeFileLists(
        args.spec, args.model, args.example, args.test, args.cts, args.log)
    '''

    # For js
    tg.FileNames.InitializeFileLists(
        args.spec, args.model, args.example, args.test, args.jsTest, args.cts, args.log)
    # end

    Configuration.force_regenerate = args.force

def NeedRegenerate():
    if not all(os.path.exists(f) for f in \
        [tg.FileNames.modelFile, tg.FileNames.exampleFile, tg.FileNames.testFile]):
        return True
    specTime = os.path.getmtime(tg.FileNames.specFile) + 10
    modelTime = os.path.getmtime(tg.FileNames.modelFile)
    exampleTime = os.path.getmtime(tg.FileNames.exampleFile)
    testTime = os.path.getmtime(tg.FileNames.testFile)
    if all(t > specTime for t in [modelTime, exampleTime, testTime]):
        return False
    return True

# For js
def NeedRegenerateForJS():
    if not os.path.exists(tg.FileNames.jsFile):
        return True
    specTime = os.path.getmtime(tg.FileNames.specFile) + 10
    jsTime = os.path.getmtime(tg.FileNames.jsFile)
    if jsTime > specTime:
        return False
    return True
# end

# Write headers for generated files, which are boilerplate codes only related to filenames
def InitializeFiles(model_fd, example_fd, test_fd):
    fileHeader = "// clang-format off\n// Generated file (from: {spec_file}). Do not edit"
    testFileHeader = """\
#include "../../TestGenerated.h"\n
namespace {spec_name} {{
// Generated {spec_name} test
#include "{example_file}"
// Generated model constructor
#include "{model_file}"
}} // namespace {spec_name}\n"""
    # This regex is to remove prefix and get relative path for #include
    pathRegex = r".*((frameworks/ml/nn/(runtime/test/)?)|(vendor/google/[a-z]*/test/))"
    specFileBase = os.path.basename(tg.FileNames.specFile)
    print(fileHeader.format(spec_file=specFileBase), file=model_fd)
    print(fileHeader.format(spec_file=specFileBase), file=example_fd)
    print(fileHeader.format(spec_file=specFileBase), file=test_fd)
    print(testFileHeader.format(
        model_file=re.sub(pathRegex, "", tg.FileNames.modelFile),
        example_file=re.sub(pathRegex, "", tg.FileNames.exampleFile),
        spec_name=tg.FileNames.specName), file=test_fd)

# Dump is_ignored function for IgnoredOutput
def DumpCtsIsIgnored(model, model_fd):
    isIgnoredTemplate = """\
inline bool {is_ignored_name}(int i) {{
  static std::set<int> ignore = {{{ignored_index}}};
  return ignore.find(i) != ignore.end();\n}}\n"""
    print(isIgnoredTemplate.format(
        ignored_index=tg.GetJointStr(model.GetIgnoredOutputs(), method=lambda x: str(x.index)),
        is_ignored_name=str(model.isIgnoredFunctionName)), file=model_fd)

# Dump Model file for Cts tests
def DumpCtsModel(model, model_fd):
    assert model.compiled
    if model.dumped:
        return
    print("void %s(Model *model) {"%(model.createFunctionName), file=model_fd)

    # Phase 0: types
    for t in model.GetTypes():
        if t.scale == 0.0 and t.zeroPoint == 0 and t.extraParams is None:
            typeDef = "OperandType %s(Type::%s, %s);"%(t, t.type, t.GetDimensionsString())
        else:
            if t.extraParams is None or t.extraParams.hide:
                typeDef = "OperandType %s(Type::%s, %s, %s, %d);"%(
                    t, t.type, t.GetDimensionsString(), tg.PrettyPrintAsFloat(t.scale), t.zeroPoint)
            else:
                typeDef = "OperandType %s(Type::%s, %s, %s, %d, %s);"%(
                    t, t.type, t.GetDimensionsString(), tg.PrettyPrintAsFloat(t.scale), t.zeroPoint,
                    t.extraParams.GetConstructor())

        IndentedPrint(typeDef, file=model_fd)

    # Phase 1: add operands
    print("  // Phase 1, operands", file=model_fd)
    for op in model.operands:
        IndentedPrint("auto %s = model->addOperand(&%s);"%(op, op.type), file=model_fd)

    # Phase 2: operations
    print("  // Phase 2, operations", file=model_fd)
    for p in model.GetParameters():
        paramDef = "static %s %s[] = %s;\nmodel->setOperandValue(%s, %s, sizeof(%s) * %d);"%(
            p.type.GetCppTypeString(), p.initializer, p.GetListInitialization(), p,
            p.initializer, p.type.GetCppTypeString(), p.type.GetNumberOfElements())
        IndentedPrint(paramDef, file=model_fd)
    for op in model.operations:
        IndentedPrint("model->addOperation(ANEURALNETWORKS_%s, {%s}, {%s});"%(
            op.optype, tg.GetJointStr(op.ins), tg.GetJointStr(op.outs)), file=model_fd)

    # Phase 3: add inputs and outputs
    print ("  // Phase 3, inputs and outputs", file=model_fd)
    IndentedPrint("model->identifyInputsAndOutputs(\n  {%s},\n  {%s});"%(
        tg.GetJointStr(model.GetInputs()), tg.GetJointStr(model.GetOutputs())), file=model_fd)

    # Phase 4: set relaxed execution if needed
    if (model.isRelaxed):
        print ("  // Phase 4: set relaxed execution", file=model_fd)
        print ("  model->relaxComputationFloat32toFloat16(true);", file=model_fd)

    print ("  assert(model->isValid());", file=model_fd)
    print ("}\n", file=model_fd)
    DumpCtsIsIgnored(model, model_fd)
    model.dumped = True

def DumpMixedType(operands, feedDict):
    supportedTensors = [
        "DIMENSIONS",
        "TENSOR_FLOAT32",
        "TENSOR_INT32",
        "TENSOR_QUANT8_ASYMM",
        "TENSOR_OEM_BYTE",
        "TENSOR_QUANT16_SYMM",
        "TENSOR_FLOAT16",
        "TENSOR_BOOL8",
        "TENSOR_QUANT8_SYMM_PER_CHANNEL",
        "TENSOR_QUANT16_ASYMM",
        "TENSOR_QUANT8_SYMM",
    ]
    typedMap = {t: [] for t in supportedTensors}
    FeedAndGet = lambda op, d: op.Feed(d).GetListInitialization()
    # group the operands by type
    for operand in operands:
        try:
            typedMap[operand.type.type].append(FeedAndGet(operand, feedDict))
            typedMap["DIMENSIONS"].append("{%d, {%s}}"%(
                operand.index, GetJointStr(operand.dimensions)))
        except KeyError as e:
            traceback.print_exc()
            sys.exit("Cannot dump tensor of type {}".format(operand.type.type))
    mixedTypeTemplate = """\
{{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> Dimensions map
  .operandDimensions = {{{dimensions_map}}},
  // int -> FLOAT32 map
  .float32Operands = {{{float32_map}}},
  // int -> INT32 map
  .int32Operands = {{{int32_map}}},
  // int -> QUANT8_ASYMM map
  .quant8AsymmOperands = {{{uint8_map}}},
  // int -> QUANT16_SYMM map
  .quant16SymmOperands = {{{int16_map}}},
  // int -> FLOAT16 map
  .float16Operands = {{{float16_map}}},
  // int -> BOOL8 map
  .bool8Operands = {{{bool8_map}}},
  // int -> QUANT8_SYMM_PER_CHANNEL map
  .quant8ChannelOperands = {{{int8_map}}},
  // int -> QUANT16_ASYMM map
  .quant16AsymmOperands = {{{uint16_map}}},
  // int -> QUANT8_SYMM map
  .quant8SymmOperands = {{{quant8_symm_map}}},
}}"""
    return mixedTypeTemplate.format(
        dimensions_map=tg.GetJointStr(typedMap.get("DIMENSIONS", [])),
        float32_map=tg.GetJointStr(typedMap.get("TENSOR_FLOAT32", [])),
        int32_map=tg.GetJointStr(typedMap.get("TENSOR_INT32", [])),
        uint8_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_ASYMM", []) +
                                 typedMap.get("TENSOR_OEM_BYTE", [])),
        int16_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT16_SYMM", [])),
        float16_map=tg.GetJointStr(typedMap.get("TENSOR_FLOAT16", [])),
        int8_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_SYMM_PER_CHANNEL", [])),
        bool8_map=tg.GetJointStr(typedMap.get("TENSOR_BOOL8", [])),
        uint16_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT16_ASYMM", [])),
        quant8_symm_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_SYMM", []))
    )

# Dump Example file for Cts tests
def DumpCtsExample(example, example_fd):
    print("std::vector<MixedTypedExample>& get_%s() {" % (example.examplesName), file=example_fd)
    print("static std::vector<MixedTypedExample> %s = {" % (example.examplesName), file=example_fd)
    for inputFeedDict, outputFeedDict in example.feedDicts:
        print ('// Begin of an example', file = example_fd)
        print ('{\n.operands = {', file = example_fd)
        inputs = DumpMixedType(example.model.GetInputs(), inputFeedDict)
        outputs = DumpMixedType(example.model.GetOutputs(), outputFeedDict)
        print ('//Input(s)\n%s,' % inputs , file = example_fd)
        print ('//Output(s)\n%s' % outputs, file = example_fd)
        print ('},', file = example_fd)
        if example.expectedMultinomialDistributionTolerance is not None:
          print ('.expectedMultinomialDistributionTolerance = %f' %
                 example.expectedMultinomialDistributionTolerance, file = example_fd)
        print ('}, // End of an example', file = example_fd)
    print("};", file=example_fd)
    print("return %s;" % (example.examplesName), file=example_fd)
    print("};\n", file=example_fd)

# Dump Test file for Cts tests
def DumpCtsTest(example, test_fd):
    testTemplate = """\
TEST_F({test_case_name}, {test_name}) {{
    execute({namespace}::{create_model_name},
            {namespace}::{is_ignored_name},
            {namespace}::get_{examples_name}(){log_file});\n}}\n"""
    if example.model.version is not None:
        testTemplate += """\
TEST_AVAILABLE_SINCE({version}, {test_name}, {namespace}::{create_model_name})\n"""
    print(testTemplate.format(
        test_case_name="DynamicOutputShapeTest" if example.model.hasDynamicOutputShape \
                       else "GeneratedTests",
        test_name=str(example.testName),
        namespace=tg.FileNames.specName,
        create_model_name=str(example.model.createFunctionName),
        is_ignored_name=str(example.model.isIgnoredFunctionName),
        examples_name=str(example.examplesName),
        version=example.model.version,
        log_file=tg.FileNames.logFile), file=test_fd)

# For js
def CheckActions(model, example):
    assert model.compiled
    if model.dumped:
        return
    # check: types
    for t in model.GetTypes():
        if t.type not in Configuration.support_types and \
           t.type not in str(Configuration.support_types).lower():
            util.WriteLineToFile("    skip not support types: %s (%s)" % (
                   example.examplesName, t.type), sys.stderr)
            return
        else :
            # use "TENSOR_FLOAT32" to support "TENSOR_FLOAT16"
            if t.type == "TENSOR_FLOAT16":
                t.type = "TENSOR_FLOAT32"

            # use "FLOAT32" to support "FLOAT16"
            if t.type == "FLOAT16":
                t.type = "FLOAT32"


    # support layout: NHWC
    for p in example.model.GetParameters():
        if p.type.type == "BOOL":
            if p.GetValueAsNumpy() == False:
                if p in model.operands:
                    model.operands.remove(p)
                for op in model.operations:
                    if p in op.ins:
                        op.ins.remove(p)
            else :
                util.WriteLineToFile("    skip not support layout: %s (%s)" % (
                       example.examplesName, p.GetValueAsNumpy()), sys.stderr)
                return

    # check data type
    for operation in model.operations:
        if operation.optype not in Configuration.check_list.keys() and \
           operation.optype not in str(Configuration.check_list.keys()).lower():
            util.WriteLineToFile("    skip not support operation code: %s (%s)" % (
                   example.examplesName, operation.optype), sys.stderr)
            return
        else :
            for inputIndex in range(len(example.model.GetInputs())):
                t = example.model.GetInputs()[inputIndex].type
                c = Configuration.check_list[operation.optype]["inputs"]
                if inputIndex in c:
                    if t.type not in c[inputIndex]["types"]:
                        util.WriteLineToFile("    skip not support input(type): %s (%s)" % (
                               example.examplesName, t.type), sys.stderr)
                        return
                    if len(t.dimensions) not in c[inputIndex]["dimensions"]:
                        util.WriteLineToFile("    skip not support input(dimension): %s (%s)" % (
                               example.examplesName, t.dimensions), sys.stderr)
                        return
                else :
                    util.WriteLineToFile("    skip not support input: %s (%s)" % (
                           example.examplesName, example.model.GetInputs()[inputIndex]), sys.stderr)
                    return

            for parameterIndex in range(len(example.model.GetParameters())):
                t = example.model.GetParameters()[parameterIndex].type
                c = Configuration.check_list[operation.optype]["inputs"]
                pii = parameterIndex + len(example.model.GetInputs())
                if pii in c:
                    if t.type not in c[pii]["types"]:
                        util.WriteLineToFile("    skip not support parameter(type): %s (%s)" % (
                               example.examplesName, t.type), sys.stderr)
                        return
                    if len(t.dimensions) not in c[pii]["dimensions"]:
                        util.WriteLineToFile("    skip not support parameter(dimension): %s (%s)" % (
                               example.examplesName, t.dimensions), sys.stderr)
                        return
                else :
                    util.WriteLineToFile("    skip not support parameter: %s (%s)" % (
                           example.examplesName, example.model.GetParameters()[parameterIndex]), sys.stderr)
                    return

            for outputIndex in range(len(example.model.GetOutputs())):
                t = example.model.GetOutputs()[outputIndex].type
                c = Configuration.check_list[operation.optype]["outputs"]
                if outputIndex in c:
                    if t.type not in c[outputIndex]["types"]:
                        util.WriteLineToFile("    skip not support output(type): %s (%s)" % (
                               example.examplesName, t.type), sys.stderr)
                        return
                    if len(t.dimensions) not in c[outputIndex]["dimensions"]:
                        util.WriteLineToFile("    skip not support output(dimension): %s (%s)" % (
                               example.examplesName, t.dimensions), sys.stderr)
                        return
                else :
                    util.WriteLineToFile("    skip not support output: %s (%s)" % (
                           example.examplesName, example.model.GetOutputs()[outputIndex]), sys.stderr)
                    return

    # check: input and output and values
    for inputFeedDict, outputFeedDict in example.feedDicts:
        for nnInOp in example.model.GetInputs():
            # check input value is None
            if len(inputFeedDict[nnInOp]) is 0:
                # For "TRANSPOSE": if perm is not given, it is set to (n-1...0)
                if model.operations[0].optype == "TRANSPOSE":
                    perm_value = []
                    perm_dimensions = []
                    for num in range(len(model.operands[0].type.dimensions)):
                        perm_value.insert(0, num)
                    # set "perm" dimensions
                    model.operands[1].type.dimensions.clear()
                    model.operands[1].type.dimensions.append(len(model.operands[0].type.dimensions))
                    # set "perm" value
                    inputFeedDict[nnInOp] = perm_value
                else :
                    util.WriteLineToFile("    skip input value is None: %s (%s - %s)" % (
                           example.examplesName, model.operations[0].optype, nnInOp), sys.stderr)
                    return

    # check: compatible dimensions
    if model.operations[0].optype == "MUL" or model.operations[0].optype == "ADD":
        if model.operands[0].type != model.operands[1].type:
            if len(model.operands[0].type.dimensions) != 1 or len(model.operands[1].type.dimensions) != 1:
                util.WriteLineToFile("    skip not support input(compatible dimensions): %s (%s - %s)" % (
                       example.examplesName, model.operands[0].type.dimensions,
                       model.operands[1].type.dimensions), sys.stderr)
                return

    # check: scale
    if model.operations[0].optype == "CONV_2D" or model.operations[0].optype == "DEPTHWISE_CONV_2D":
        if model.operands[0].type.type == "TENSOR_QUANT8_ASYMM":
            if example.model.GetOutputs()[0].type.scale <= (
               model.operands[0].type.scale * model.operands[1].type.scale):
                util.WriteLineToFile("    skip not support output(scale): %s (%s <= (%s * %s))" % (
                       example.examplesName, example.model.GetOutputs()[0].type.scale,
                       model.operands[0].type.scale, model.operands[1].type.scale), sys.stderr)
                return

def DumpJSTest(model, example, js_fd):
    CheckActions(model, example)

    androidNNOpInputList = example.model.GetInputs()
    androidNNOpOutputList = example.model.GetOutputs()
    androidNNOpParamList = example.model.GetParameters()

    androidNNOpType = model.operations[0].optype
    opInfoDict = util.OperationsInfoDict[androidNNOpType]
    equalOpType = opInfoDict['equalOp'] # for Web NN API
    bBiase = opInfoDict['paramList'].count('bias') != 0
    bActivation = opInfoDict['paramList'].count('activation') != 0

    if bActivation:
        activationParam = androidNNOpParamList[-1]
        if activationParam.value[0] == util.ReluType.RELU.value:
            bActivation = True # add relu opertation to model graph
        else:
            bActivation = False

    # TODO
    Configuration.example_count += 1

    testPurpose = ""
    testIndex = ""
    testPurposeArray = tg.FileNames.specName.split('_')
    equalOp = util.OperationsInfoDict[androidNNOpType]['equalOp']

    if testPurposeArray[-1].isdigit():
        if testPurposeArray[-2] is not None and str(testPurposeArray[-2]) != 'v1':
            testPurpose = '%s (%s)' % (equalOp, ' '.join(testPurposeArray[1:-1]))
            testIndex = testPurposeArray[-1]
        else :
            testPurposeArray[-2] = '%s_%s' % (testPurposeArray[-2] ,testPurposeArray[-1])
            testPurpose = '%s (%s)' % (equalOp, ' '.join(testPurposeArray[1:-1]))
    else:
        if len(testPurposeArray) > 1:
            testPurpose = '%s (%s)' % (equalOp, ' '.join(testPurposeArray[1:]))
        else:
            testPurpose = equalOp

    util.WriteLineToFile("", js_fd)

    if bBiase:
        testPurpose = testPurpose + " + add"

    if bActivation:
         testPurpose = testPurpose + " + relu"

    for inputFeedDict, outputFeedDict in example.feedDicts:
        if Configuration.single_example_flag:
            if testIndex == "":
                util.WriteLineToFile("  it('%s example', async function() {" % testPurpose, js_fd)
            else:
                util.WriteLineToFile("  it('%s example/%s', async function() {" % (
                       testPurpose, testIndex), js_fd)
        else:
            if testIndex == "":
                util.WriteLineToFile("  it('%s example-%s', async function() {" % (
                       testPurpose, Configuration.example_count), js_fd)
            else:
                util.WriteLineToFile("  it('%s example/%s-%s', async function() {" % (
                       testPurpose, testIndex, Configuration.example_count), js_fd)


    insList = model.operations[0].ins
    # remove dulicate item
    # ["op1", "op2", "op3", "pad0", "pad0", "pad0", "pad0", "stride", "stride", "act"] -> ["op1", "op2", "op3", "pad0","stride", "act"]
    actualIns = list(dict.fromkeys(insList))
    for op in actualIns:
        # 1. add oprand into model graph
        # if op is model input, call nn.input to create oprand
        # elif op is model constant, call nn.constant to create oprand
        # else op just is paramter for opertation, define variable for nn.<operation> function
        bFilterOp = util.CheckFilterOp(androidNNOpType, insList.index(op))
        if op in androidNNOpInputList:
            util.WriteLineToFile("    const %s = nn.input('%s', %s);" % (op, op, util.GetOperandDesc(op.type, bFilterOp)), js_fd)
        else:
            if op.type.type.startswith('TENSOR_'):
                util.WriteLineToFile("    const %s = nn.constant(%s, new %s(%s));" % (op, util.GetOperandDesc(op.type, bFilterOp), util.TypedArrayTypeMapping[op.type.type].value, util.reorderValues(op.value, op.type)), js_fd)
            else:
                if insList.index(op) == len(insList) - 1:
                    if bActivation:
                        util.WriteLineToFile('    const %s = %s;' % (op, androidNNOpParamList[androidNNOpParamList.index(op)].value[0]), js_fd)
                else:
                    util.WriteLineToFile('    const %s = %s;' % (op, androidNNOpParamList[androidNNOpParamList.index(op)].value[0]), js_fd)


    outputOp = androidNNOpOutputList[0]

    # add opertation(s) into model graph
    if androidNNOpType == 'ADD':
        # nn.add doesn't fuse relu activation, only has two params
        addParamList = [ins.name for ins in insList[:-1]]
        if not bActivation:
            util.WriteLineToFile("    const %s = nn.%s(%s);" % (outputOp, equalOp, ', '.join(addParamList)), js_fd)
        else:
            util.WriteLineToFile("    const intermediateOutput = nn.%s(%s);" % (equalOp, ', '.join(addParamList)), js_fd)
            util.WriteLineToFile("    const %s = nn.relu(intermediateOutput);" % outputOp, js_fd)
    elif androidNNOpType == 'CONV_2D':
        paddingParam = [str(insList[5]), str(insList[6]), str(insList[3]), str(insList[4])]
        strideParam = [str(insList[8]), str(insList[7])]
        conv2dParam = "%s, %s, [%s], [%s], [1, 1], 1, 'nhwc'" % (insList[0].name, insList[1].name, ', '.join(paddingParam), ', '.join(strideParam))
        util.WriteLineToFile("    const intermediateOutput1 = nn.%s(%s);" % (equalOp, conv2dParam), js_fd)
        biasOp = insList[2]
        if not bActivation:
            util.WriteLineToFile("    const %s = nn.add(intermediateOutput1, %s);" % (outputOp, biasOp), js_fd)
        else:
            util.WriteLineToFile("    const intermediateOutput2 = nn.add(intermediateOutput1, %s);" % biasOp, js_fd)
            util.WriteLineToFile("    const %s = nn.relu(intermediateOutput2);" % outputOp, js_fd)

    util.WriteLineToFile("    const model = await nn.createModel([{name: '%s', operand: %s}]);" % (outputOp, outputOp), js_fd)

    # compiling model
    util.WriteLineToFile("    const compilation = await model.createCompilation();", js_fd)

    # executing model
    util.WriteLineToFile("    const execution = await compilation.createExecution();", js_fd)

    # set input
    for nnInOp in androidNNOpInputList:
        values = inputFeedDict[nnInOp]
        bFilterOp = util.CheckFilterOp(androidNNOpType, insList.index(nnInOp))
        if bFilterOp:
            values = util.reorderValues(values, nnInOp.type)
        util.WriteLineToFile("    execution.setInput('%s', new %s(%s));" % (nnInOp, util.TypedArrayTypeMapping[nnInOp.type.type].value, values), js_fd)

    # set output
    util.WriteLineToFile("    const expected = %s;" % outputFeedDict[outputOp], js_fd)
    util.WriteLineToFile("    const outputBuffer = new %s(expected.length);" % util.TypedArrayTypeMapping[outputOp.type.type].value, js_fd)
    util.WriteLineToFile("    execution.setOutput('%s', outputBuffer);" % outputOp, js_fd)

    util.WriteLineToFile("    await execution.startCompute();", js_fd)

    # assert output
    util.WriteLineToFile("    checkOutput(outputBuffer, expected);", js_fd)

    util.WriteLineToFile("  });", js_fd)

    model.dumped = True
# end

if __name__ == '__main__':
    ParseCmdLine()

    while tg.FileNames.NextFile():
        ''' Original
        if Configuration.force_regenerate or NeedRegenerate():
        '''

        # For js
        if Configuration.force_regenerate or NeedRegenerateForJS():
        # end
            print ("--Generating test(s) from spec: %s" % tg.FileNames.specFile, file = sys.stderr)
            exec(open(tg.FileNames.specFile, "r").read())

            ''' Original
            print("Output CTS model: %s" % tg.FileNames.modelFile, file=sys.stderr)
            print("Output example:%s" % tg.FileNames.exampleFile, file=sys.stderr)
            print("Output CTS test: %s" % tg.FileNames.testFile, file=sys.stderr)
            with SmartOpen(tg.FileNames.modelFile) as model_fd, \
                 SmartOpen(tg.FileNames.exampleFile) as example_fd, \
                 SmartOpen(tg.FileNames.testFile) as test_fd:
                InitializeFiles(model_fd, example_fd, test_fd)
                Example.DumpAllExamples(
                    DumpModel=DumpCtsModel, model_fd=model_fd,
                    DumpExample=DumpCtsExample, example_fd=example_fd,
                    DumpTest=DumpCtsTest, test_fd=test_fd)
            '''

            # For js
            with SmartOpen(tg.FileNames.modelFile) as model_fd, \
                 SmartOpen(tg.FileNames.exampleFile) as example_fd, \
                 SmartOpen(tg.FileNames.testFile) as test_fd, \
                 SmartOpen(tg.FileNames.jsFile) as js_fd:
                util.WriteMochaTestCaseHeads(js_fd)
                Example.DumpAllExamples(
                    DumpModel=DumpCtsModel, model_fd=model_fd,
                    DumpExample=DumpCtsExample, example_fd=example_fd,
                    DumpTest=DumpCtsTest, test_fd=test_fd,
                    DumpJS=DumpJSTest, js_fd=js_fd)
                print ("});", file = js_fd)
            # end
        else:
            print ("Skip file: %s" % tg.FileNames.specFile, file = sys.stderr)

        if Configuration.example_count == 0:
            os.remove(tg.FileNames.jsFile)
            print (">>Remove empty JS CTS test: %s\n" % tg.FileNames.jsFile, file = sys.stderr)
        else :
            print (">>Output JS CTS test: %s\n" % tg.FileNames.jsFile, file = sys.stderr)
        ''' Original
        with SmartOpen(tg.FileNames.ctsFile, mode="a") as cts_fd:
            print("#include \"../generated/tests/%s.cpp\""%os.path.basename(tg.FileNames.specFile),
                file=cts_fd)
        '''
