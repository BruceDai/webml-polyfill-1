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
    util.WriteLineToFile(fileHeader.format(spec_file=specFileBase), model_fd)
    util.WriteLineToFile(fileHeader.format(spec_file=specFileBase), example_fd)
    util.WriteLineToFile(fileHeader.format(spec_file=specFileBase), test_fd)
    util.WriteLineToFile(testFileHeader.format(
        model_file=re.sub(pathRegex, "", tg.FileNames.modelFile),
        example_file=re.sub(pathRegex, "", tg.FileNames.exampleFile),
        spec_name=tg.FileNames.specName), test_fd)

# Dump is_ignored function for IgnoredOutput
def DumpCtsIsIgnored(model, model_fd):
    isIgnoredTemplate = """\
inline bool {is_ignored_name}(int i) {{
  static std::set<int> ignore = {{{ignored_index}}};
  return ignore.find(i) != ignore.end();\n}}\n"""
    util.WriteLineToFile(isIgnoredTemplate.format(
        ignored_index=tg.GetJointStr(model.GetIgnoredOutputs(), method=lambda x: str(x.index)),
        is_ignored_name=str(model.isIgnoredFunctionName)), model_fd)

# Dump Model file for Cts tests
def DumpCtsModel(model, model_fd):
    assert model.compiled
    if model.dumped:
        return
    util.WriteLineToFile("void %s(Model *model) {"%(model.createFunctionName), model_fd)

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
    util.WriteLineToFile("  // Phase 1, operands", model_fd)
    for op in model.operands:
        IndentedPrint("auto %s = model->addOperand(&%s);"%(op, op.type), file=model_fd)

    # Phase 2: operations
    util.WriteLineToFile("  // Phase 2, operations", model_fd)
    for p in model.GetParameters():
        paramDef = "static %s %s[] = %s;\nmodel->setOperandValue(%s, %s, sizeof(%s) * %d);"%(
            p.type.GetCppTypeString(), p.initializer, p.GetListInitialization(), p,
            p.initializer, p.type.GetCppTypeString(), p.type.GetNumberOfElements())
        IndentedPrint(paramDef, file=model_fd)
    for op in model.operations:
        IndentedPrint("model->addOperation(ANEURALNETWORKS_%s, {%s}, {%s});"%(
            op.optype, tg.GetJointStr(op.ins), tg.GetJointStr(op.outs)), file=model_fd)

    # Phase 3: add inputs and outputs
    util.WriteLineToFile("  // Phase 3, inputs and outputs", model_fd)
    IndentedPrint("model->identifyInputsAndOutputs(\n  {%s},\n  {%s});"%(
        tg.GetJointStr(model.GetInputs()), tg.GetJointStr(model.GetOutputs())), file=model_fd)

    # Phase 4: set relaxed execution if needed
    if (model.isRelaxed):
        util.WriteLineToFile("  // Phase 4: set relaxed execution", model_fd)
        util.WriteLineToFile("  model->relaxComputationFloat32toFloat16(true);", model_fd)

    util.WriteLineToFile("  assert(model->isValid());", model_fd)
    util.WriteLineToFile("}\n", model_fd)
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
    util.WriteLineToFile("std::vector<MixedTypedExample>& get_%s() {" % (example.examplesName), example_fd)
    util.WriteLineToFile("static std::vector<MixedTypedExample> %s = {" % (example.examplesName), example_fd)
    for inputFeedDict, outputFeedDict in example.feedDicts:
        util.WriteLineToFile('// Begin of an example', example_fd)
        util.WriteLineToFile('{\n.operands = {', example_fd)
        inputs = DumpMixedType(example.model.GetInputs(), inputFeedDict)
        outputs = DumpMixedType(example.model.GetOutputs(), outputFeedDict)
        util.WriteLineToFile('//Input(s)\n%s,' % inputs , example_fd)
        util.WriteLineToFile('//Output(s)\n%s' % outputs, example_fd)
        util.WriteLineToFile('},', example_fd)
        if example.expectedMultinomialDistributionTolerance is not None:
          util.WriteLineToFile('.expectedMultinomialDistributionTolerance = %f' %
                 example.expectedMultinomialDistributionTolerance, example_fd)
        util.WriteLineToFile('}, // End of an example', example_fd)
    util.WriteLineToFile("};", example_fd)
    util.WriteLineToFile("return %s;" % (example.examplesName), example_fd)
    util.WriteLineToFile("};\n", example_fd)

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
    util.WriteLineToFile(testTemplate.format(
        test_case_name="DynamicOutputShapeTest" if example.model.hasDynamicOutputShape \
                       else "GeneratedTests",
        test_name=str(example.testName),
        namespace=tg.FileNames.specName,
        create_model_name=str(example.model.createFunctionName),
        is_ignored_name=str(example.model.isIgnoredFunctionName),
        examples_name=str(example.examplesName),
        version=example.model.version,
        log_file=tg.FileNames.logFile), test_fd)


def DumpJSTest(model, example, js_fd):
    assert model.compiled
    if model.dumped:
        return

    # TODO: Check Relu Supported

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

    '''
    # select specifying type models
    select_specifying_flag = False
    if model.operations[0].optype in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
        if model.operands[0].type.type == "TENSOR_QUANT8_ASYMM_SIGNED" and \
           model.operands[1].type.type == "TENSOR_QUANT8_SYMM_PER_CHANNEL":
            select_specifying_flag = True

    if not select_specifying_flag:
        util.WriteLineToFile("    skip not select types: %s" % example.examplesName, sys.stderr)
        return
    '''

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

    # set js test names
    Configuration.example_count += 1

    test_name = ""
    test_index = ""
    # args = "options"
    per_channel_types = dict()
    test_info = tg.FileNames.specName.capitalize().replace("_", " ")
    test_name_array = test_info.split(" ")

    if test_name_array[-1].isdigit():
        if test_name_array[-2] is not None and str(test_name_array[-2]) != "v1":
            test_name = " ".join(test_name_array[:-1])
            test_index = test_name_array[-1]
        else :
            test_name = " ".join(test_name_array[:-1])
            test_name = str(test_name) + "_" + test_name_array[-1]
    else:
        test_name = test_info

    util.WriteLineToFile("", js_fd)

    # TODO it description of plusing relu / add 
    for inputFeedDict, outputFeedDict in example.feedDicts:
        if Configuration.single_example_flag:
            if test_index == "":
                util.WriteLineToFile("  it('check result for %s example', async function() {" % test_name, js_fd)
            else:
                util.WriteLineToFile("  it('check result for %s example/%s', async function() {" % (
                       test_name, test_index), js_fd)
        else:
            if test_index == "":
                util.WriteLineToFile("  it('check result for %s example-%s', async function() {" % (
                       test_name, Configuration.example_count), js_fd)
            else:
                util.WriteLineToFile("  it('check result for %s example/%s-%s', async function() {" % (
                       test_name, test_index, Configuration.example_count), js_fd)

        # create input
        parametersV2List = []

        for nnInOp in example.model.GetInputs():
            parametersV2List.append(nnInOp.name)
            util.WriteLineToFile("    const %s = nn.input('%s', %s);" % (nnInOp, nnInOp, util.GetOperandDesc(nnInOp.type)), js_fd)

        # only for options with one output
        # invoke nn.<operations_name> function
        nnOutOp = example.model.GetOutputs()[0]
        androidNNOpType = model.operations[0].optype
        webNNOpType = util.OperationsMapping[androidNNOpType].value

        androidPramaterList = example.model.GetParameters()
        androidParametersLen = len(androidPramaterList)
        # Compare with normal value(nomarLen), if androidParametersLen > nomarLen, then the front 0, 1, ... shoud be nn.constant
        #then rest paramters map to Web NN paramters.

        print(androidPramaterList)
        print('parameter len %d' % androidParametersLen)

        # 'pramsLen': {
        #     'explicit': 7,
        #     'implicit': 4

        if androidNNOpType == 'ADD':
            util.WriteLineToFile("    const %s = nn.%s(%s);" % (nnOutOp, webNNOpType, ', '.join(parametersV2List)), js_fd)
        elif androidNNOpType == 'CONV_2D':
            # print(androidPramaterList[0].value)
            # print(inputFeedDict)
            if androidParametersLen == util.OperationsParmLen.CONV_2D_IMPLICIT + 2:
                # count in filter and bias, using nn.constant to create filter and bias
                filterOp = androidPramaterList[0]
                parametersV2List.append(filterOp.name)
                util.WriteConstantOprandLine(filterOp, js_fd, True)

                # padding
                if androidPramaterList[2].value[0] == util.PaddingCode.VALID:
                    padding = [0, 0, 0, 0]
                else:
                    # TODO padding SAME
                    pass
                parametersV2List.append('padding')
                util.WriteLineToFile('    const padding = %s;' % padding, js_fd)

                # strides
                strides = [androidPramaterList[4].value[0], androidPramaterList[3].value[0]] # [stride_height, stride_width]
                parametersV2List.append('strides')
                util.WriteLineToFile('    const strides = %s;' % strides, js_fd)

                # dilations
                dilations = [1, 1]
                parametersV2List.append('dilations')
                util.WriteLineToFile('    const dilations = %s;' % dilations, js_fd)

                # groups
                groups = 1
                parametersV2List.append('groups')
                util.WriteLineToFile('    const groups = %d;' % groups, js_fd)

                # layout
                layout = 'nhwc'
                parametersV2List.append('layout')
                util.WriteLineToFile("    const layout = '%s';" % layout, js_fd)

                # nn.conv2d
                util.WriteLineToFile('    const intermediateOutput = nn.conv2d(%s);' % ', '.join(parametersV2List), js_fd)

                # plus nn.add for computing with bias
                biasOp = androidPramaterList[1]
                util.WriteConstantOprandLine(biasOp, js_fd)
                util.WriteLineToFile('    const %s = nn.add(intermediateOutput, %s);' % (nnOutOp, biasOp), js_fd)
                                                             
            elif androidParametersLen == util.OperationsParmLen.CONV_2D_EXPLICIT + 2:
                # count in filter and bias, using nn.constant to create filter and bias
                filterOp = androidPramaterList[0]
                parametersV2List.append(filterOp.name)
                util.WriteConstantOprandLine(filterOp, js_fd, True)
                biasOp = androidPramaterList[1]
                parametersV2List.append(biasOp.name)
                util.WriteConstantOprandLine(biasOp, js_fd)

                # explicit padding
                padding = [androidPramaterList[2].value for index in range(2, 6)]


        # TODO check relu type of model.GetParameters() case (operation) by case (operation)
        # if FUSED_NONE, skip add relu  
        # else FUSED_RELU, need add relu code

        # outputsNames = example.model.GetOutputs()
        # outputList = ["{name: '%s', operand: %s}" % (n, n) for n in outputsNames]
        # util.WriteLineToFile("    const model = await nn.createModel([%s]);" % ','.join(outputList), js_fd)

        util.WriteLineToFile("    const model = await nn.createModel([{name: '%s', operand: %s}]);" % (nnOutOp, nnOutOp), js_fd)

        # compiling model
        util.WriteLineToFile("    const compilation = await model.createCompilation();", js_fd)

        # executing model
        util.WriteLineToFile("    const execution = await compilation.createExecution();", js_fd)

        # set input
        for nnInOp in example.model.GetInputs():
            util.WriteLineToFile("    execution.setInput('%s', new %s(%s));" % (nnInOp, util.TypedArrayTypeMapping[nnInOp.type.type].value, inputFeedDict[nnInOp]), js_fd)

        # set output
        util.WriteLineToFile("    const expected = %s;" % outputFeedDict[nnOutOp], js_fd)
        util.WriteLineToFile("    const outputBuffer = new %s(expected.length);" % util.TypedArrayTypeMapping[nnOutOp.type.type].value, js_fd)
        util.WriteLineToFile("    execution.setOutput('%s', outputBuffer);" % nnOutOp, js_fd)

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
            util.WriteLineToFile("--Generating test(s) from spec: %s" % tg.FileNames.specFile, sys.stderr)
            exec(open(tg.FileNames.specFile, "r").read())

            ''' Original
            util.WriteLineToFile("Output CTS model: %s" % tg.FileNames.modelFile, sys.stderr)
            util.WriteLineToFile("Output example:%s" % tg.FileNames.exampleFile, sys.stderr)
            util.WriteLineToFile("Output CTS test: %s" % tg.FileNames.testFile, sys.stderr)
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
                util.WriteLineToFile("});", js_fd)
            # end
        else:
            util.WriteLineToFile("Skip file: %s" % tg.FileNames.specFile, sys.stderr)

        if Configuration.example_count == 0:
            os.remove(tg.FileNames.jsFile)
            util.WriteLineToFile(">>Remove empty JS CTS test: %s\n" % tg.FileNames.jsFile, sys.stderr)
        else :
            util.WriteLineToFile(">>Output JS CTS test: %s\n" % tg.FileNames.jsFile, sys.stderr)
        ''' Original
        with SmartOpen(tg.FileNames.ctsFile, mode="a") as cts_fd:
            util.WriteLineToFile("#include \"../generated/tests/%s.cpp\""%os.path.basename(tg.FileNames.specFile),
                file=cts_fd)
        '''
