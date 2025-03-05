# !type: ignore
import os
import re
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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
model = model.dense_head
print(model)
# model.dense_head.exporting = True
backbone_dense_name = model.__class__.__name__

input = {"spatial_features_2d": torch.randn(1, 384, 64, 64).cuda()}

camel2snake = lambda s: re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()
torch.onnx.export(
    model,
    (input, {}),
    f"./deploy_files/{camel2snake(backbone_dense_name)}.onnx",
    input_names=["spatial_features_2d"],
    output_names=["rois", "roi_score", "roi_label"],
    dynamic_axes={
        "spatial_features_2d": {0: "batch_size"},
        "rois": {0: "batch_size"},
        "roi_score": {0: "batch_size"},
        "roi_label": {0: "batch_size"},
    },
    opset_version=14,
    verbose=True,
)
