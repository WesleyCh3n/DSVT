import argparse
import re
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def get_logger(base: Path):
    return common_utils.create_logger(base / "log_trt.log", rank=0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path", type=Path)
    parser.add_argument("ckpt", help="checkpoint file path")
    parser.add_argument(
        "-o",
        "--outputbase",
        help="output base dir",
        default=Path("./deploy_files/"),
        type=Path,
    )
    parser.add_argument("-3", "--backbone3d", action="store_true")
    parser.add_argument("-2", "--backbone2d", action="store_true")
    parser.add_argument("-d", "--densehead", action="store_true")
    parser.add_argument("-D", "--debug", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


####### DSVT #######
class AllDSVTBlocksTRT(nn.Module):
    def __init__(self, dsvtblocks_list, layer_norms_list):
        super().__init__()
        self.layer_norms_list = layer_norms_list
        self.dsvtblocks_list = dsvtblocks_list

    def forward(
        self,
        pillar_features,
        set_voxel_inds_tensor_shift_0,
        set_voxel_inds_tensor_shift_1,
        set_voxel_masks_tensor_shift_0,
        set_voxel_masks_tensor_shift_1,
        pos_embed_tensor,
    ):
        outputs = pillar_features

        residual = outputs
        blc_id = 0
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)

        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 1
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)

        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 2
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)

        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 3
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id : set_id + 1].squeeze(0)
        pos_embed = (
            pos_embed_tensor[blc_id : blc_id + 1, set_id : set_id + 1]
            .squeeze(0)
            .squeeze(0)
        )
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)

        outputs = self.layer_norms_list[blc_id](residual + outputs)

        return outputs


def camel2snake(s):
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()


def dense_head_2onnx(model, output_base: Path):
    export_path = output_base / f"{camel2snake(model.__class__.__name__)}.onnx"

    input = {
        "spatial_features_2d": torch.randn(
            1,
            cfg["MODEL"].BACKBONE_3D.dim_feedforward[0],
            *cfg["MODEL"].BACKBONE_3D.output_shape,
        ).cuda()
    }
    input_names = ["spatial_features_2d"]
    output_names = ["rois", "roi_score", "roi_label"]
    dynamic_axes = {
        "spatial_features_2d": {0: "batch_size"},
        "rois": {0: "batch_size"},
        "roi_score": {0: "batch_size"},
        "roi_label": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (input, {}),
        export_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        verbose=True,
    )


def backbone_2d_2onnx(model, output_base: Path):
    export_path = output_base / f"{camel2snake(model.__class__.__name__)}.onnx"
    input = {
        "spatial_features": torch.randn(
            1, *[cfg["MODEL"].MAP_TO_BEV.NUM_BEV_FEATURES for _ in range(3)]
        ).cuda()
    }
    input_names = ["spatial_features"]
    output_names = ["spatial_features_2d"]
    dynamic_axes = {
        "spatial_features": {0: "batch_size"},
        "spatial_features_2d": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        (input, {}),
        export_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        verbose=True,
    )


def backbone_3d_2onnx(model, output_base: Path):
    batch_dict = torch.load("path to batch_dict.pth", map_location="cuda")  # TODO:
    inputs = batch_dict
    with torch.no_grad():
        DSVT_Backbone = model
        dsvtblocks_list = DSVT_Backbone.stage_0
        layer_norms_list = DSVT_Backbone.residual_norm_stage_0
        inputs = model.vfe(inputs)
        voxel_info = DSVT_Backbone.input_layer(inputs)
        set_voxel_inds_list = [
            [voxel_info[f"set_voxel_inds_stage{s}_shift{i}"] for i in range(2)]
            for s in range(1)
        ]
        set_voxel_masks_list = [
            [voxel_info[f"set_voxel_mask_stage{s}_shift{i}"] for i in range(2)]
            for s in range(1)
        ]
        pos_embed_list = [
            [
                [voxel_info[f"pos_embed_stage{s}_block{b}_shift{i}"] for i in range(2)]
                for b in range(4)
            ]
            for s in range(1)
        ]

        pillar_features = inputs["voxel_features"]
        alldsvtblockstrt_inputs = (
            pillar_features,
            set_voxel_inds_list[0][0],
            set_voxel_inds_list[0][1],
            set_voxel_masks_list[0][0],
            set_voxel_masks_list[0][1],
            torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0),
        )

        input_names = [
            "src",
            "set_voxel_inds_tensor_shift_0",
            "set_voxel_inds_tensor_shift_1",
            "set_voxel_masks_tensor_shift_0",
            "set_voxel_masks_tensor_shift_1",
            "pos_embed_tensor",
        ]
        output_names = [
            "output",
        ]
        dynamic_axes = {
            "src": {
                0: "voxel_number",
            },
            "set_voxel_inds_tensor_shift_0": {
                1: "set_number_shift_0",
            },
            "set_voxel_inds_tensor_shift_1": {
                1: "set_number_shift_1",
            },
            "set_voxel_masks_tensor_shift_0": {
                1: "set_number_shift_0",
            },
            "set_voxel_masks_tensor_shift_1": {
                1: "set_number_shift_1",
            },
            "pos_embed_tensor": {
                2: "voxel_number",
            },
            "output": {
                0: "voxel_number",
            },
        }

        onnx_path = output_base / "dsvt.onnx"

        allptransblocktrt = (
            AllDSVTBlocksTRT(dsvtblocks_list, layer_norms_list).eval().cuda()
        )
        torch.onnx.export(
            allptransblocktrt,
            alldsvtblockstrt_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )


def main():
    args = parse_arguments()
    cfg_file = args.config
    cfg_from_yaml_file(cfg_file, cfg)
    logger = get_logger(args.outputbase)
    args.outputbase.mkdir(parents=True, exist_ok=True)
    if args.debug:
        logger.info("Enable Debug mode")

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg["DATA_CONFIG"],
        class_names=cfg["CLASS_NAMES"],
        batch_size=1,
        dist=False,
        workers=8,
        logger=logger,
        training=False,
    )
    model = build_network(
        model_cfg=cfg["MODEL"], num_class=len(cfg["CLASS_NAMES"]), dataset=test_set
    )
    ckpt = args.ckpt
    if not args.debug:
        model.load_params_from_file(
            filename=ckpt, logger=logger, to_cpu=False, pre_trained_path=None
        )
    model.eval().cuda()
    if args.backbone2d:
        backbone_2d_2onnx(model.backbone_2d, args.outputbase)
    if args.densehead:
        dense_head_2onnx(model.dense_head, args.outputbase)

    logger.info("Finished")


if __name__ == "__main__":
    main()


####### torch to trt engine #######
# trtexec --onnx={path to onnx} --saveEngine={path to save trtengine} \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=1 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=src:3000x192,set_voxel_inds_tensor_shift_0:2x170x36,set_voxel_inds_tensor_shift_1:2x100x36,set_voxel_masks_tensor_shift_0:2x170x36,set_voxel_masks_tensor_shift_1:2x100x36,pos_embed_tensor:4x2x3000x192 \
# --optShapes=src:20000x192,set_voxel_inds_tensor_shift_0:2x1000x36,set_voxel_inds_tensor_shift_1:2x700x36,set_voxel_masks_tensor_shift_0:2x1000x36,set_voxel_masks_tensor_shift_1:2x700x36,pos_embed_tensor:4x2x20000x192 \
# --maxShapes=src:35000x192,set_voxel_inds_tensor_shift_0:2x1500x36,set_voxel_inds_tensor_shift_1:2x1200x36,set_voxel_masks_tensor_shift_0:2x1500x36,set_voxel_masks_tensor_shift_1:2x1200x36,pos_embed_tensor:4x2x35000x192 \
####### torch to trt engine #######
