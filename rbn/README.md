# [Representative Batch Normalization (RBN) with Feature Calibration](#contents)

The official implementation of the CVPR2021 oral paper: Representative Batch Normalization with Feature Calibration

**You only need to replace the BN with our RBN without any other adjustment.**

## Description

Batch Normalization (BatchNorm) has become the default component in modern neural networks to stabilize
training. In BatchNorm, centering and scaling operations,
along with mean and variance statistics, are utilized for
feature standardization over the batch dimension. The
batch dependency of BatchNorm enables stable training
and better representation of the network, while inevitably
ignores the representation differences among instances. We
propose to add a simple yet effective feature calibration
scheme into the centering and scaling operations of BatchNorm, enhancing the instance-specific representations with
the negligible computational cost. The centering calibration strengthens informative features and reduces noisy features. The scaling calibration restricts the feature intensity to form a more stable feature distribution. Our proposed variant of BatchNorm, namely Representative BatchNorm, can be plugged into existing methods to boost the
performance of various tasks such as classification, detection, and segmentation.

**In this repo, we use RBN layer to replace BN layer in ResNet.**

# [Model Architecture](#contents)

The overall network architecture of ResNet is show below:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

# [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：60,000 32*32 colorful images in 10 classes
    - Train：50,000 images
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```bash
└─dataset
    ├─ilsvrc                # train dataset
    └─validation_preprocess # evaluate dataset
```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

- Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# infer example
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

# gpu benchmark example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)
```

- Running on CPU

```bash
# standalone training example
python train.py --device_target=CPU --data_path=[DATASET_PATH]  --config_path [CONFIG_PATH] --pre_trained=[CHECKPOINT_PATH](optional)

# infer example
python eval.py --data_path=[DATASET_PATH] --checkpoint_file_path=[CHECKPOINT_PATH] --config_path [CONFIG_PATH]  --device_target=CPU
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the code directory to "/path/resnet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the code directory to "/path/resnet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──resnet
  ├── README.md
  ├── config                               # parameter configuration
    ├── resnet18_cifar10_config.yaml
    ├── resnet18_cifar10_config_gpu.yaml
    ├── resnet18_imagenet2012_config.yaml
    ├── resnet18_imagenet2012_config_gpu.yaml
    ├── resnet34_imagenet2012_config.yaml
    ├── resnet50_cifar10_config.yaml
    ├── resnet50_imagenet2012_Boost_config.yaml     # High performance version: The performance is improved by more than 10% and the precision decrease less than 1%
    ├── resnet50_imagenet2012_Ascend_Thor_config.yaml
    ├── resnet50_imagenet2012_config.yaml
    ├── resnet50_imagenet2012_GPU_Thor_config.yaml
    ├── resnet101_imagenet2012_config.yaml
    ├── resnet152_imagenet2012_config.yaml
    └── se-resnet50_imagenet2012_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_parameter_server_train.sh      # launch ascend parameter server training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_parameter_server_train_gpu.sh  # launch gpu parameter server training(8 pcs)
    ├── run_eval_gpu.sh                    # launch gpu evaluation
    ├── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    ├── run_gpu_resnet_benchmark.sh        # launch gpu benchmark train for resnet50 with imagenet2012
    |── run_eval_gpu_resnet_benckmark.sh   # launch gpu benchmark eval for resnet50 with imagenet2012
    └── cache_util.sh                      # a collection of helper functions to manage cache
  ├── src
    ├── dataset.py                         # data preprocessing
    ├─  eval_callback.py                   # evaluation callback while training
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── resnet.py                          # resnet backbone, including resnet50 and resnet101 and se-resnet50
    └── resnet_gpu_benchmark.py            # resnet50 for GPU benchmark
    ├── model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       ├──moxing_adapter.py                # moxing adapter
  ├── export.py                            # export model for inference
  ├── mindspore_hub_conf.py                # mindspore hub interface
  ├── eval.py                              # eval net
  ├── train.py                             # train net
  └── gpu_resent_benchmark.py              # GPU benchmark for resnet50
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for ResNet18 and ResNet50, CIFAR-10 dataset

```bash
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "poly"           # decay mode can be selected in steps, ploy and default
"lr_init": 0.01,                  # initial learning rate
"lr_end": 0.00001,                # final learning rate
"lr_max": 0.1,                    # maximum learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for ResNet18 and ResNet50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 256,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "Linear",        # decay mode for generating learning rate
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.8,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for ResNet34, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 256,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 1,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"optimizer": 'Momentum',          # optimizer
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 1.0,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
```

- Config for ResNet101, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 120,                # epoch size for training
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for ResNet152, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 140,                # epoch size for training
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_path":"./",      # the save path of the checkpoint relative to the execution path
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "steps"          # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1,                        # base learning rate
"lr_end": 0.0001,                 # end learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for SE-ResNet50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 28 ,                # epoch size for creating learning rate
"train_epoch_size": 24            # actual train epoch size
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 4,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 3,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr_init": 0.0,                   # initial learning rate
"lr_max": 0.3,                    # maximum learning rate
"lr_end": 0.0001,                 # end learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

#### Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# infer example
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
[CONFIG_PATH]

# gpu benchmark training example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)

# gpu benchmark infer example
bash run_eval_gpu_resnet_benchmark.sh [DATASET_PATH] [CKPT_PATH] [BATCH_SIZE](optional) [DTYPE](optional)
```

For distributed training, a hostfile configuration needs to be created in advance.

Please follow the instructions in the link [GPU-Multi-Host](https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_gpu.html).

#### Running parameter server mode training

- Parameter server training Ascend example

```bash
bash run_parameter_server_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)
```

- Parameter server training GPU example

```bash
bash run_parameter_server_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)
```

#### Evaluation while training

```bash
# evaluation with distributed training Ascend example:
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with standalone training Ascend example:
bash run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with distributed training GPU example:
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with standalone training GPU example:
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)
```

`RUN_EVAL` and `EVAL_DATASET_PATH` are optional arguments, setting `RUN_EVAL`=True allows you to do evaluation while training. When `RUN_EVAL` is set, `EVAL_DATASET_PATH` must also be set.
And you can also set these optional arguments: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` for python script when `RUN_EVAL` is True.

By default, a standalone cache server would be started to cache all eval images in tensor format in memory to improve the evaluation performance. Please make sure the dataset fits in memory (Around 30GB of memory required for ImageNet2012 eval dataset, 6GB of memory required for CIFAR-10 eval dataset).

Users can choose to shutdown the cache server after training or leave it alone for future usage.

## [Resume Process](#contents)

### Usage

#### Running on Ascend

```text
# distributed training
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]

# standalone training
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]
```

### Result

- Training ResNet18 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.5783054
epoch: 2 step: 195, loss is 1.0682616
epoch: 3 step: 195, loss is 0.8836588
epoch: 4 step: 195, loss is 0.36090446
epoch: 5 step: 195, loss is 0.80853784
...
```

- Training ResNet18 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 625, loss is 4.757934
epoch: 2 step: 625, loss is 4.0891967
epoch: 3 step: 625, loss is 3.9131956
epoch: 4 step: 625, loss is 3.5302577
epoch: 5 step: 625, loss is 3.597817
...
```

- Training ResNet34 with ImageNet2012 dataset

```text
# 分布式训练结果（8P）
epoch: 2 step: 625, loss is 4.181185
epoch: 3 step: 625, loss is 3.8856044
epoch: 4 step: 625, loss is 3.423355
epoch: 5 step: 625, loss is 3.506971
...
```

- Training ResNet50 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 195, loss is 1.9601055
epoch: 2 step: 195, loss is 1.8555021
epoch: 3 step: 195, loss is 1.6707983
epoch: 4 step: 195, loss is 1.8162166
epoch: 5 step: 195, loss is 1.393667
...
```

- Training ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
...
```

- Training ResNet101 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.805483
epoch: 2 step: 5004, loss is 3.2121816
epoch: 3 step: 5004, loss is 3.429647
epoch: 4 step: 5004, loss is 3.3667371
epoch: 5 step: 5004, loss is 3.1718972
...
```

- Training ResNet152 with ImageNet2012 dataset

```bash
# 分布式训练结果（8P）
epoch: 1 step: 5004, loss is 4.184874
epoch: 2 step: 5004, loss is 4.013571
epoch: 3 step: 5004, loss is 3.695777
epoch: 4 step: 5004, loss is 3.3244863
epoch: 5 step: 5004, loss is 3.4899402
...
```

- Training SE-ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 5.1779146
epoch: 2 step: 5004, loss is 4.139395
epoch: 3 step: 5004, loss is 3.9240637
epoch: 4 step: 5004, loss is 3.5011306
epoch: 5 step: 5004, loss is 3.3501816
...
```

- GPU Benchmark of ResNet50 with ImageNet2012 dataset

```bash
# ========START RESNET50 GPU BENCHMARK========
epoch: [0/1] step: [20/5004], loss is 6.940182 Epoch time: 12416.098 ms, fps: 412 img/sec.
epoch: [0/1] step: [40/5004], loss is 7.078993Epoch time: 3438.972 ms, fps: 1488 img/sec.
epoch: [0/1] step: [60/5004], loss is 7.559594Epoch time: 3431.516 ms, fps: 1492 img/sec.
epoch: [0/1] step: [80/5004], loss is 6.920937Epoch time: 3435.777 ms, fps: 1490 img/sec.
epoch: [0/1] step: [100/5004], loss is 6.814013Epoch time: 3437.154 ms, fps: 1489 img/sec.
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
# evaluation
Usage: bash run_eval.sh [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH]
```

```bash
# evaluation example
bash run_eval.sh resnet50 cifar10 ~/cifar10-10-verify-bin ~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt --config_path /.yaml
```

> checkpoint can be produced in training process.

#### Running on GPU

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Evaluating ResNet18 with CIFAR-10 dataset

```bash
result: {'acc': 0.9363061543521088} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet18 with ImageNet2012 dataset

```bash
result: {'acc': 0.7053685897435897} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet50 with CIFAR-10 dataset

```bash
result: {'acc': 0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet50 with ImageNet2012 dataset

```bash
result: {'acc': 0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet34 with ImageNet2012 dataset

```bash
result: {'top_1_accuracy': 0.736758814102564} ckpt=train_parallel0/resnet-90_625.ckpt
```

- Evaluating ResNet101 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9429417413572343, 'top_1_accuracy': 0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

- Evaluating ResNet152 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9438420294494239, 'top_1_accuracy': 0.78817221518} ckpt= resnet152-140_5004.ckpt
```

- Evaluating SE-ResNet50 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9342589628681178, 'top_1_accuracy': 0.768065781049936} ckpt=train_parallel0/resnet-24_5004.ckpt

```

## Inference Process

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

The checkpoint_file_path parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./resnet'" on default_config.yaml file.
#          Set "file_format='AIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./resnet'" on the website UI interface.
#          Add "file_format='AIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/resnet" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space, otherwise the process will be killed for execeeding memory limits.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET]  [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
```

- `NET_TYPE` can choose from [resnet18, se-resnet50, resnet34, resnet50, resnet101, resnet152].
- `DATASET` can choose from [cifar10, imagenet].
- `DEVICE_ID` is optional, default value is 0.

