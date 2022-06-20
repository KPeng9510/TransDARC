# TransDARC (ReadME in progress)


## Extract and save features
Please first train the video feature extraction backbone from video swin transformer in mmaction2 repo using drive and act dataset, 

In our paper, we use the same configuration of the video swin transformer for swin-base as mentioned in https://github.com/SwinTransformer/Video-Swin-Transformer.git while using the ImageNet pretraining.

## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```



