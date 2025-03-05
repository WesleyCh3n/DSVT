# type: ignore
import os

import torch
import torch.onnx

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# load model
cfg_file = "./cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml"
cfg_from_yaml_file(cfg_file, cfg)
if os.path.exists("./deploy_files") == False:
    os.mkdir("./deploy_files")
log_file = "./deploy_files/log_trt.log"
logger = common_utils.create_logger(log_file, rank=0)
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    workers=8,
    logger=logger,
    training=False,
)


model = build_network(
    model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set
)
model.eval().cuda()
print(model.backbone_2d)

torch.onnx.export(
    model.backbone_2d,
    (
        {
            "spatial_features": torch.randn(
                1,
                cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES,
                *cfg.MODEL.MAP_TO_BEV.INPUT_SHAPE[:2],
            ).cuda()
        },
        {},
    ),
    "./deploy_files/bev_backbone.onnx",
    input_names=["spatial_features"],
    output_names=["spatial_features_2d"],
    dynamic_axes={
        "spatial_features": {0: "batch_size"},
        "spatial_features_2d": {0: "batch_size"},
    },
    opset_version=14,
    verbose=True,
)
