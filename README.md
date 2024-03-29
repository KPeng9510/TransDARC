# TransDARC (IROS 2022, ReadME in progress)

## News
**TransDARC** [[**PDF**](https://arxiv.org/pdf/2203.00927.pdf)] is accepted to **IROS2022** for an **Oral** presentation.

## Extract and save features
Please first train the video feature extraction backbone from video swin transformer in mmaction2 repo using drive and act dataset, 

In our paper, we use the same configuration of the video swin transformer for swin-base as mentioned in https://github.com/SwinTransformer/Video-Swin-Transformer.git while using the ImageNet pretraining.

The corresponding configuration of traing for swin_base_patch244_window877_kinetics400_22k.py 



train_pipeline = [


    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=2),
    
    
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label'])
    
    
]


val_pipeline = [


    dict(
    
    
        type='SampleFrames',
        
        
        clip_len=32,
        
        
        frame_interval=2,
        
        
        num_clips=2,
        
        
        test_mode=True),
        
        
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label'])
    
    
]


test_pipeline = [


    dict(
    
    
        type='SampleFrames',
        
        
        clip_len=32,
        
        
        frame_interval=2,
        
        
        num_clips=2,
        
        
        test_mode=True),
        
        
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label'])
    
    
]

PLease average the clips output at the top of video swin transformer

## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```
## Verification of our code

The logits of Video Swin Base are available at https://drive.google.com/drive/folders/1MJY8toH3PSV--pA2EvL8qjPIjcaTdmvr?usp=sharing, which is for fine-grained level split 0 driver activity recognition. (notice that, the result reported in the paper is the mean average over three splits, i.e., split0, split1, and split2)

Please note that the performance is evaluated unfortunately differently by using unbalanced mean average Top-1 accuracy. Under balanced accuracy our model can still achieve 71.0 accuracy for fine-grained activity recognition on the test set. In case you want to compare with us, you could either evaluate following the same way our paper use and compare with TransDARC, or contact my email to get more results for the other tasks considering balanced accuracy evaluation. Thanks!

## Please consider citing our paper once you are interested in it. [[**PDF**](https://arxiv.org/pdf/2203.00927.pdf)]

```
@article{peng2022transdarc,
  title={TransDARC: Transformer-based Driver Activity Recognition with Latent Space Feature Calibration},
  author={Peng, Kunyu and Roitberg, Alina and Yang, Kailun and Zhang, Jiaming and Stiefelhagen, Rainer},
  journal={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022}
}
```

