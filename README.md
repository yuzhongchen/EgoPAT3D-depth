# Depth Estimation for EgoPAT3Dv2


## Abstract
We use depth estimation techniques to improve the possibility of improving the performance of the EgoPAT3Dv2  model, so we examined the performance for several depth estimators. To improve the ability of EgoPAT3Dv2 , the best depth estimation algorithm is trained and tested on EgoPAT3Dv2 model by setting the depth of the estimated motion target to be the depth at the 2D position of the motion target in the predicted depth map.

## Environment setup
This code has been tested on Ubuntu 20.04, Python 3.7.0, Pytorch 2.0.0, CUDA 11.2. Detailed environment is in requirement.txt
Please install related libraries before running this code. You can e-mail yc6317@nyu.edu to require singularity overlay

## Train
Please use train_DDP_depth.py --config_file ./configs/baseline_rgb_convnext_t.yaml to run the file. A sample training script is saved in baseline_rgb_manual_train.SBATCH
Please updating setting in your config file following the instruction in cfg_defaults.py. The supervised model is trained in current config.

### Prepare datasets
Please refer to [here]([https://ai4ce.github.io/EgoPAT3Dv2/]) for EgoPAT3Dv2 dataset and model. If you have any problem, please contact yc6317@nyu.edu.

#### Dataset folder hierarchy
Please modify the DATA.DATAROOT to be the path to your data root.

## Test and Validate
Download the pre-trained model [here]([https://drive.google.com/file/d/119Fap67qfxIt1AsCme0ABD3ZjW4c4EIa/view?usp=sharing](https://drive.google.com/drive/folders/1QTOZMX6zNO8-WMuHXZt4h5CXXnv_sewR?usp=sharing)) and set the checkpoints directory.
The checkpoint.pth is the weight for EgoPAT3Dv2 model. encoder.pth is for depth encoder. depth.pth is for depth decoder. The pose.pth and pose_encoder is for pose decoder and pose encoder, but they are not used for testing.

For depth estimator
```
python test_DDP_depth.py --model_epoch baseline_rgb_convnext_t/epoch_<your epoch> \
 --checkpoint <your path to weight file> --config_file ./configs/baseline_rgb_convnext_t.yaml > \
 ./experiment/baseline_rgb_convnext_t/eval/output_logs/<your epoch>.log 2>&1
```


For motion target prediction
```
python test_DDP_fast_depth.py --model_epoch baseline_rgb_convnext_t/epoch_<your epoch> \
 --checkpoint <your path to weight file> --config_file ./configs/baseline_rgb_convnext_t.yaml > \
 ./experiment/baseline_rgb_convnext_t/eval/output_logs/<your epoch>.log 2>&1
```

```
python my_eval.py --model_name baseline_rgb_convnext_t
```

The testing result will be printed at last for depth estimators.
The testing result will be saved in the `./results/model_name` directory for motion target prediction.


## Evaluation
Evaluation for motion target predictors.

```
python my_eval.py --model_name baseline_rgb_convnext_t
```

The results and visualization results will be saved in the `./results/` directory.

