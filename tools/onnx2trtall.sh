#!/bin/bash

for file in $(find deploy_files -name "*.onnx" -not -name "*_fold.onnx" -print); do
  polygraphy surgeon sanitize $file --fold-constants -o deploy_files/$(basename -s .onnx $file)_fold.onnx
done

onnx2trt() {
  ./onnx2trt.sh $1 "${1%.*}.engine" $2 $3 $4
}

onnx2trt "deploy_files/dynamic_pillar_vfe_3d_fold.onnx" \
    "points.1:1000x6" \
    "points.1:2000x6" \
    "points.1:3000x6"

onnx2trt ./deploy_files/dsvt_fold.onnx \
    "src:3000x192,set_voxel_inds_tensor_shift_0:2x170x36,set_voxel_inds_tensor_shift_1:2x100x36,set_voxel_masks_tensor_shift_0:2x170x36,set_voxel_masks_tensor_shift_1:2x100x36,pos_embed_tensor:4x2x3000x192" \
    "src:20000x192,set_voxel_inds_tensor_shift_0:2x1000x36,set_voxel_inds_tensor_shift_1:2x700x36,set_voxel_masks_tensor_shift_0:2x1000x36,set_voxel_masks_tensor_shift_1:2x700x36,pos_embed_tensor:4x2x20000x192" \
    "src:35000x192,set_voxel_inds_tensor_shift_0:2x1500x36,set_voxel_inds_tensor_shift_1:2x1200x36,set_voxel_masks_tensor_shift_0:2x1500x36,set_voxel_masks_tensor_shift_1:2x1200x36,pos_embed_tensor:4x2x35000x192"

onnx2trt ./deploy_files/point_pillar_scatter3d_fold.onnx \
    "pillar_features:1000x192,voxel_coords:1000x4" \
    "pillar_features:2000x192,voxel_coords:2000x4" \
    "pillar_features:3000x192,voxel_coords:3000x4"

onnx2trt ./deploy_files/base_bevres_backbone_fold.onnx \
    "spatial_features:10x192x192x192" \
    "spatial_features:20x192x192x192" \
    "spatial_features:30x192x192x192"

onnx2trt ./deploy_files/center_head_fold.onnx \
    "spatial_features_2d:1x384x468x468" \
    "spatial_features_2d:2x384x468x468" \
    "spatial_features_2d:3x384x468x468"
