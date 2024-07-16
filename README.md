# STARS ✨
[[Project Page]](https://soroushmehraban.github.io/stars)


This is an official PyTorch implementation of **STARS: Self-supervised Tuning for 3D Action Recognition in Skeleton Sequences**. 

## Data Preparation ⚙️
Follow the same steps explained in [MAMP](https://github.com/maoyunyao/MAMP) repository and put the result in `./data` directory. After data preparation, the following data structure is expected:
```
.
└── data/
    ├── NTU60_XSub.npz
    ├── NTU60_XView.npz
    ├── NTU120_XSub.npz
    ├── NTU120_XSet.npz
    └── PKUv2_XSub.npz
```

## Self-supervised Pre-Training 📉
### First stage
First stage is trained using the [MAMP method](https://github.com/maoyunyao/MAMP). You can refer to their repository and download the model weights. Create a new directory in root directory of project called `pretrained_weights` and place their weights by adding `mamp` at the beginning of each file. After doing this step it's expected to have the following file structure:
```
.
├── config
├── data
├── ...
└── pretrained_weights/
    ├── mamp_ntu60_xsub.pth
    ├── mamp_ntu60_xview.pth
    ├── mamp_ntu120_xsub.pth
    └── mamp_ntu120_xview.pth
```
**Note**: Since they did not provide weights for pkl experiment, we retrained using our own systems. That's why the result we report on paper is different than what they report for a fair comparison.
### Second stage
You can tune the weights of MAMP by running the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                             --nnodes=1 
                             --nproc_per_node=4 
                             main.py \
                             pretrain_clrtuning \
                             --config config/ntu60/pretext/clrtuning_xsub_merged.yaml \
                             --use-wandb \
                             --wandb-name CLRTuning-XSub
```
In the example above, `0,1,2,3` are the GPU IDs since we used 4 GPUs (Change it to 0 if you have a single GPU) and `--nproc_per_node=4` is the number of GPUs. Additionally, since we use wandb for logging, we set `--use-wandb` to upload the logs to server and `--wandb-name` as for the Run ID.

You can also run for the XView evaluation by changing the config to `clrtuning_xview_second.yaml`. Upon completion, go inside the `work_dir` and place the best ckpt to `pretrained_weights` with the naming expected in config file of third stage.

Alternatively, you can download the weights from [here](https://drive.google.com/drive/folders/1HKhL1rGRbx_PH79JNzA78y-49eYd4I82?usp=sharing)

## Tuning and Evaluation 📊
For all the evaluations, we use the same training code as the MAMP for a fair comparison. We have placed the modified version of their repository here under `MAMP` directory. You can run the evaluations similiar to what stated in the provided bash files. e.g. you can run the semi-supervised training for NTU-60 XView as follows:
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main_finetune.py \
        --config ./config/ntu60_xview_joint/semi_0.1_t120_layer8_decay.yaml \
        --output_dir ./output_dir/ntu60_xview/semi1_1/ \
        --resume  ./output_dir/ntu60_xview/semi1_1/checkpoint-last.pth \
        --log_dir ./output_dir/ntu60_xview/semi1_1_log/ \
        --finetune ../pretrained_weights/clrtuning_thirdstage_ntu60_view.pt \
        --dist_eval \
        --lr 3e-4 \
        --min_lr 1e-5
```
Notes:
- In the config `yaml` files, `data_path` values are started with `/path/to/` that you need to replace them with place where data is located.
- resume in command above is for the case that your code is preempted in the middle and you want to resume from last checkpoint. If not exists, it starts from the beginning.

### K-NN and Few-shot evaluation
For K-NN evaluation, you can do it as follows:
```
python3 knn_eval.py \
        --data-path <PATH-TO-DATA> \
        --checkpoint <PATH-TO-MODEL-WEIGHT>
```
For the case of Few-shot evaluation, you need to download the data from [here](https://drive.google.com/file/d/18epVXRSXkHnBYE1ZKghfeKPSTzqZtqcf/view?usp=sharing). The data consists of all NTU-120 XSub samples except those that are already present in NTU-60 XSub. For checkpoint you have to use the weight of NTU-60 XSub model.

## Results 📜
You can find the pretrained weights [here](https://drive.google.com/drive/folders/1xIEo8ZVBAb3QyvNsWxMXb9hDPDa8Fjay?usp=sharing).
| Protocols | NTU-60 X-sub | NTU-60 X-view | NTU-120 X-sub | NTU-120 X-set |
|:---------:|:------------:|:-------------:|:-------------:|:-------------:|
|  Linear   |     87.1     |      90.9     |      79.9     |      80.8     |
| KNN (k=1) |     79.9     |      88.6     |      67.6     |      67.7     |
| Finetune  |     93.0     |      97.5     |      89.9     |      91.4     |


## Acknowledgement 🙏
The framework of our code is extended from the following repositories. We sincerely thank the authors for releasing the codes.
- The framework of our code is based on [ActCLR](https://github.com/LanglandsLin/ActCLR).
- The encoder is based on [MAMP](https://github.com/maoyunyao/MAMP).

## Licence

This project is licensed under the terms of the MIT license.