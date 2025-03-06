# Deployment

## Torch to ONNX

The script will export torch model to ONNX format.
Currently, available modules include `vfe:DynPillarVFE3D`, `backbone3d:DSVT`,
 `map2bev:PointPillarScatter3d`, `backbone2d:BaseBEVResBackbone`, `densehead:CenterHead`.

- Usage

```
usage: deploy.py [-h] [-o OUTPUTBASE] [-a] [-v] [-3] [-m] [-f] [-2] [-d] [-p] [-r] [--skip-ckpt] config ckpt

positional arguments:
  config                config file path
  ckpt                  checkpoint file path

options:
  -h, --help            show this help message and exit
  -o OUTPUTBASE, --outputbase OUTPUTBASE
                        output base dir
  -a, --all
  -v, --vfe
  -3, --backbone3d
  -m, --map2bev
  -f, --pfe
  -2, --backbone2d
  -d, --densehead
  -p, --pointhead
  -r, --roithead
  --skip-ckpt           export barebone only for testing
```

- Example: export backbone 2D

```
python deploy.py -2 ./cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml ./checkpoint.pth
```

- Example: export dense head

```
python deploy.py -d ./cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml ./checkpoint.pth
```

## ONNX to TensorRT

- Convert single
```
./onnx2trt {path to onnx} {path to trt engine}
   input_1:[batch]x[shape0]x[shape1],input_2:[batch]x[shape0]x[shape1] # min shape
   input_1:[batch]x[shape0]x[shape1],input_2:[batch]x[shape0]x[shape1] # opt shape
   input_1:[batch]x[shape0]x[shape1],input_2:[batch]x[shape0]x[shape1] # max shape
```

- To convert all including `polygraphy sanitize`(fold constant),
```
./onnx2trtall.sh
```

