import argparse
import re
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def get_logger(base: Path):
    return common_utils.create_logger(base / "deploy.log", rank=0)


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
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-v", "--vfe", action="store_true")
    parser.add_argument("-3", "--backbone3d", action="store_true")
    parser.add_argument("-m", "--map2bev", action="store_true")
    parser.add_argument("-f", "--pfe", action="store_true")
    parser.add_argument("-2", "--backbone2d", action="store_true")
    parser.add_argument("-d", "--densehead", action="store_true")
    parser.add_argument("-p", "--pointhead", action="store_true")
    parser.add_argument("-r", "--roithead", action="store_true")
    parser.add_argument(
        "--skip-ckpt", action="store_true", help="export barebone only for testing"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg_file = args.config
    cfg_from_yaml_file(cfg_file, cfg)
    logger = get_logger(args.outputbase)
    args.outputbase.mkdir(parents=True, exist_ok=True)
    if args.skip_ckpt:
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
    if not args.skip_ckpt:
        model.load_params_from_file(
            filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=None
        )
    model.eval().cuda()

    if args.all:
        pass
    else:
        if args.vfe:
            model.vfe.export_onnx({"points": torch.randn(1, 6).cuda()}, args.outputbase)
        if args.backbone3d:
            model.backbone_3d.export_onnx(
                model.vfe({"points": torch.randn(1, 6).cuda()}), args.outputbase
            )
        if args.map2bev:  # pointpillarscatter3d
            model.map_to_bev_module.export_onnx(
                {
                    "pillar_features": torch.randn(
                        1, cfg["MODEL"].MAP_TO_BEV.NUM_BEV_FEATURES
                    ).cuda(),
                    "voxel_coords": torch.randn(1, 4).cuda(),
                },
                args.outputbase,
            )
        if args.pfe:
            raise NotImplemented
        if args.backbone2d:
            model.backbone_2d.export_onnx(
                {
                    "spatial_features": torch.randn(
                        1, *[cfg["MODEL"].MAP_TO_BEV.NUM_BEV_FEATURES for _ in range(3)]
                    ).cuda()
                },
                args.outputbase,
            )
        if args.densehead:
            model.dense_head.export_onnx(
                {
                    "spatial_features_2d": torch.randn(
                        1,
                        cfg["MODEL"].BACKBONE_3D.dim_feedforward[0],
                        *cfg["MODEL"].BACKBONE_3D.output_shape,
                    ).cuda()
                },
                args.outputbase,
            )

        if args.pointhead:
            raise NotImplemented
        if args.roithead:
            raise NotImplemented

    logger.info(
        "\033[94m"
        + """
use `./onnx2trt.sh` to convert onnx to TensorRT engine file.
Please adjust batch size and input dimension base on your deployed condition.

For example:
# Print usage.
./onnx2trt.sh

# 3d backbone
./onnx2trt.sh {path to onnx} {path to output trt engine} \\
    src:3000x192,set_voxel_inds_tensor_shift_0:2x170x36,set_voxel_inds_tensor_shift_1:2x100x36,set_voxel_masks_tensor_shift_0:2x170x36,set_voxel_masks_tensor_shift_1:2x100x36,pos_embed_tensor:4x2x3000x192 \\
    src:20000x192,set_voxel_inds_tensor_shift_0:2x1000x36,set_voxel_inds_tensor_shift_1:2x700x36,set_voxel_masks_tensor_shift_0:2x1000x36,set_voxel_masks_tensor_shift_1:2x700x36,pos_embed_tensor:4x2x20000x192 \\
    src:35000x192,set_voxel_inds_tensor_shift_0:2x1500x36,set_voxel_inds_tensor_shift_1:2x1200x36,set_voxel_masks_tensor_shift_0:2x1500x36,set_voxel_masks_tensor_shift_1:2x1200x36,pos_embed_tensor:4x2x35000x192

# 2d backbone
./onnx2trt.sh {path to onnx} {path to output trt engine} \\
    spatial_features_2d:1x192x192x192 \\
    spatial_features_2d:5x192x192x192 \\
    spatial_features_2d:10x192x192x192

# dense head
./onnx2trt.sh {path to onnx} {path to output trt engine} \\
    spatial_features_2d:1x384x468x468 \\
    spatial_features_2d:5x384x468x468 \\
    spatial_features_2d:10x384x468x468
"""
        + "\033[0m"
    )
    logger.info("Finished!!")


if __name__ == "__main__":
    main()

# polygraphy surgeon sanitize deploy_files/dynamic_pillar_vfe_3d.onnx --fold-constants -o deploy_files/dynamic_pillar_vfe_3d_fold.onnx
