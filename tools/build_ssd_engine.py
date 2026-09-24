#!/usr/bin/env python3
# build_ssd_engine.py -- build ssd_mobilenet_v2_coco.engine for THIS machine's TensorRT.
#
# Run inside the JetBot docker container:
#     cd /workspace/ssd_build && bash try_orders.sh
# or for a single attempt:
#     python3 build_ssd_engine.py --input-order 0,2,1
#
# ASCII output only: the container locale is not UTF-8, printing non-ASCII crashes.
#
# WHY THIS EXISTS
# ---------------
# The prebuilt ssd_mobilenet_v2_coco.engine shipped by NVIDIA was compiled for
# JetPack 4.3 / TensorRT 6. A serialized TensorRT engine is NOT portable across
# TensorRT versions -- loading it on JetPack 4.5 / TensorRT 7.1.3 fails with
# "Serialization Error in verifyHeader: Version tag does not match".
# So the engine has to be rebuilt on the target machine.
#
# TWO DEVIATIONS FROM jetbot's OWN ssd_pipeline_to_uff()
# ------------------------------------------------------
# 1. jetbot re-exports the checkpoint with the TF Object Detection API, which is
#    not installed in the JetBot container. We skip that and use the
#    frozen_inference_graph.pb that already ships in the model zoo tarball, and
#    read the handful of numbers we need out of pipeline.config as plain text.
# 2. jetbot hardcodes inputOrder=[1, 2, 0] and passes scoreConverter="SIGMOID".
#    Both are TensorRT 6 era. On TensorRT 7, scoreConverter is not a field of
#    NMS_TRT (harmless INFO), and the input order differs, which makes the NMS
#    plugin abort with "#assertionnmsPlugin.cpp,246" during the engine build.
#    That assertion is a C++ abort -- it cannot be caught from Python, which is
#    why try_orders.sh runs this script once per permutation.
#
# The final step still uses jetbot's ssd_uff_to_engine(), so the engine keeps the
# tensor names ('input' / 'nms') that jetbot's ObjectDetector expects, and the
# notebook needs no changes.

from __future__ import print_function

import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, 'ssd_mobilenet_v2_coco_2018_03_29')
CONFIG = os.path.join(MODEL_DIR, 'pipeline.config')
FROZEN = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')
OUT = os.path.join(HERE, 'ssd_mobilenet_v2_coco.engine')


def bail(msg):
    print('ERROR: ' + msg)
    sys.exit(1)


# ---------- plain-text pipeline.config reader ----------

def _block(name, text):
    """Return the body of `name { ... }` by brace matching."""
    marker = name + ' {'
    i = text.find(marker)
    if i < 0:
        bail('block not found in pipeline.config: ' + name)
    j = i + len(marker)
    depth, k = 1, j
    while depth and k < len(text):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
        k += 1
    return text[j:k - 1]


_NUM = r'[-+0-9.eE]+'


def _one(key, text):
    m = re.search(r'\b' + key + r'\s*:\s*(' + _NUM + r')', text)
    if not m:
        bail('key not found in pipeline.config: ' + key)
    return float(m.group(1))


def _many(key, text):
    return [float(v) for v in
            re.findall(r'\b' + key + r'\s*:\s*(' + _NUM + r')', text)]


def feature_map_shapes(width):
    """Same rule as jetbot's _get_feature_map_shape: ceil(w/16), then halve x6."""
    fms, curr = [], int(math.ceil(width / 16.0))
    for _ in range(6):
        fms.append(curr)
        curr = int(math.ceil(curr / 2.0))
    return fms


# ---------- graph surgery straight from the frozen graph ----------

def build_uff(input_order):
    import graphsurgeon as gs
    import tensorflow as tf
    import uff
    from jetbot.ssd_tensorrt import TRT_INPUT_NAME, TRT_OUTPUT_NAME

    if not os.path.isfile(FROZEN):
        bail('missing ' + FROZEN)
    if not os.path.isfile(CONFIG):
        bail('missing ' + CONFIG)

    text = open(CONFIG, 'r').read()
    ssd = _block('ssd', text)

    num_classes = int(_one('num_classes', ssd))
    resizer = _block('fixed_shape_resizer', ssd)
    height = int(_one('height', resizer))
    width = int(_one('width', resizer))

    anchor = _block('ssd_anchor_generator', ssd)
    min_scale = _one('min_scale', anchor)
    max_scale = _one('max_scale', anchor)
    aspect_ratios = _many('aspect_ratios', anchor)
    num_layers = int(_one('num_layers', anchor))

    coder = _block('faster_rcnn_box_coder', ssd)
    y_scale = _one('y_scale', coder)
    x_scale = _one('x_scale', coder)
    height_scale = _one('height_scale', coder)
    width_scale = _one('width_scale', coder)

    nms_cfg = _block('batch_non_max_suppression', ssd)
    score_threshold = _one('score_threshold', nms_cfg)
    iou_threshold = _one('iou_threshold', nms_cfg)
    max_per_class = int(_one('max_detections_per_class', nms_cfg))
    max_total = int(_one('max_total_detections', nms_cfg))

    fms = feature_map_shapes(width)
    print('  input       : %dx%d' % (width, height))
    print('  classes     : %d (+1 background)' % num_classes)
    print('  layers      : %d, feature maps %s' % (num_layers, fms))
    print('  aspect      : %s' % aspect_ratios)
    print('  nms         : score>%g iou=%g topK=%d keepTopK=%d'
          % (score_threshold, iou_threshold, max_per_class, max_total))
    print('  inputOrder  : %s   <-- the value being tested' % (input_order,))

    graph = gs.DynamicGraph(FROZEN)
    graph.forward_inputs(graph.find_nodes_by_op('Identity'))

    input_plugin = gs.create_plugin_node(
        name=TRT_INPUT_NAME, op='Placeholder',
        dtype=tf.float32, shape=[1, height, width, 3])

    priorbox_plugin = gs.create_plugin_node(
        name='priorbox', op='GridAnchor_TRT',
        minSize=min_scale, maxSize=max_scale,
        aspectRatios=aspect_ratios,
        variance=[1.0 / y_scale, 1.0 / x_scale,
                  1.0 / height_scale, 1.0 / width_scale],
        featureMapShapes=fms,
        numLayers=num_layers)

    # scoreConverter is deliberately NOT passed: it is not a field of NMS_TRT in
    # TensorRT 7. confSigmoid=1 is what actually selects the sigmoid converter.
    nms_plugin = gs.create_plugin_node(
        name=TRT_OUTPUT_NAME, op='NMS_TRT',
        shareLocation=1, varianceEncodedInTarget=0, backgroundLabelId=0,
        confidenceThreshold=score_threshold, nmsThreshold=iou_threshold,
        topK=max_per_class, keepTopK=max_total,
        numClasses=num_classes + 1,
        inputOrder=input_order, confSigmoid=1, isNormalized=1,
        codeType=3)

    priorbox_concat = gs.create_node(
        'priorbox_concat', op='ConcatV2', dtype=tf.float32, axis=2)
    boxloc_concat = gs.create_plugin_node(
        'boxloc_concat', op='FlattenConcat_TRT_jetbot', dtype=tf.float32)
    boxconf_concat = gs.create_plugin_node(
        'boxconf_concat', op='FlattenConcat_TRT_jetbot', dtype=tf.float32)

    graph.collapse_namespaces({
        'MultipleGridAnchorGenerator': priorbox_plugin,
        'Postprocessor': nms_plugin,
        'Preprocessor': input_plugin,
        'ToFloat': input_plugin,
        'image_tensor': input_plugin,
        'Concatenate': priorbox_concat,
        'concat': boxloc_concat,
        'concat_1': boxconf_concat,
    })

    nms_node = graph.find_nodes_by_op('NMS_TRT')[0]
    print('  nms inputs  : %s' % list(nms_node.input))
    for i, name in enumerate(nms_node.input):
        if TRT_INPUT_NAME in name:
            nms_node.input.pop(i)
            break

    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)

    return uff.from_tensorflow(graph.as_graph_def(), [TRT_OUTPUT_NAME])


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-order', default='1,0,2',
                    help='comma separated permutation of 0,1,2')
    args = ap.parse_args()
    input_order = [int(v) for v in args.input_order.split(',')]
    if sorted(input_order) != [0, 1, 2]:
        bail('--input-order must be a permutation of 0,1,2')

    if not os.path.isdir(MODEL_DIR):
        bail('model folder not found: ' + MODEL_DIR)

    import tensorrt as trt
    from jetbot.ssd_tensorrt import ssd_uff_to_engine

    print('=' * 60)
    print('TensorRT %s   inputOrder %s' % (trt.__version__, input_order))
    print('=' * 60)

    print('[1/3] converting to UFF ...')
    uff_buffer = build_uff(input_order)
    print('      UFF size: %.1f MB' % (len(uff_buffer) / 1048576.0))

    print('')
    print('[2/3] building engine (a bad inputOrder aborts here) ...')
    engine = ssd_uff_to_engine(uff_buffer)
    if engine is None:
        bail('engine build returned None')

    with open(OUT, 'wb') as f:
        f.write(engine.serialize())
    print('      wrote %s (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1048576.0))

    print('')
    print('[3/3] verifying it loads back ...')
    from jetbot import ObjectDetector
    ObjectDetector(OUT)
    print('      ObjectDetector loaded the engine OK')

    print('')
    print('=' * 60)
    print('SUCCESS with inputOrder = %s' % (input_order,))
    print('Copy it next to the notebook:')
    print('  cp %s /workspace/jetbot/notebooks/object_following/' % OUT)
    print('=' * 60)


if __name__ == '__main__':
    main()
