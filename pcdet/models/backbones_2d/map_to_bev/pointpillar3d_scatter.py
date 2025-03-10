import re

import torch
import torch.nn as nn


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = (
            self.model_cfg.NUM_BEV_FEATURES // self.nz
        )

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = (
            batch_dict["pillar_features"],
            batch_dict["voxel_coords"],
        )

        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            )

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = (
                this_coords[:, 1] * self.ny * self.nx
                + this_coords[:, 2] * self.nx
                + this_coords[:, 3]
            )
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            batch_size,
            self.num_bev_features_before_compression * self.nz,
            self.ny,
            self.nx,
        )
        batch_dict["spatial_features"] = batch_spatial_features
        print(batch_dict.keys())
        return batch_dict

    def export_onnx(self, inputs, output_base):
        camel2snake = lambda s: re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()
        export_path = output_base / f"{camel2snake(self.__class__.__name__)}.onnx"
        input_names = ["pillar_features", "voxel_coords"]
        output_names = ["spatial_features"]
        dynamic_axes = {
            "pillar_features": {0: "batch_size"},
            "voxel_coords": {0: "batch_size"},
            "spatial_features": {0: "batch_size"},
        }
        torch.onnx.export(
            self,
            (inputs, {}),
            export_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            verbose=True,
        )
