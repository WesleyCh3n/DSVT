#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 <onnx> <trt> <min_shape> <opt_shape> <max_shape>"
  echo "Example:
   $0 {path to onnx} {path to trtengine}
     src:3000x192,set_voxel_inds_tensor_shift_0:2x170x36,...
     src:20000x192,set_voxel_inds_tensor_shift_0:2x1000x36,...
     src:35000x192,set_voxel_inds_tensor_shift_0:2x1500x36,..."
  exit
fi

ONNX_MODEL="$1"
TRT_MODEL="$2"
MIN_SHAPE="$3"
OPT_SHAPE="$4"
MAX_SHAPE="$5"

echo "PWD: $PWD"
echo "ONNX_MODEL: $ONNX_MODEL"
echo "TRT_MODEL: $TRT_MODEL"
echo "MIN_SHAPE: $MIN_SHAPE"
echo "OPT_SHAPE: $OPT_SHAPE"
echo "MAX_SHAPE: $MAX_SHAPE"

docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:25.01-py3 trtexec \
  --memPoolSize=workspace:4096 --verbose --device=0 --fp16 \
  --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
  --onnx=$ONNX_MODEL --saveEngine=$TRT_MODEL \
  --profilingVerbosity=detailed \
  --exportLayerInfo=${ONNX_MODEL%.*}.graph.json \
  --exportTimes=${ONNX_MODEL%.*}.timing.json \
  --exportProfile=${ONNX_MODEL%.*}.profile.json \
  --minShapes=$MIN_SHAPE \
  --optShapes=$OPT_SHAPE \
  --maxShapes=$MAX_SHAPE
