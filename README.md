# TransDARC (ReadME in progress)


## Extract and save features
Please first train the video feature extraction backbone from video swin transformer in mmaction2 repo using drive and act dataset, 

In our paper, we use the same configuration of the video swin transformer for swin-base as mentioned in https://github.com/SwinTransformer/Video-Swin-Transformer.git while using the ImageNet pretraining.

The corresponding configuration of traing for swin_base_patch244_window877_kinetics400_22k.py 


train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=2),
    
    
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label', 'fast_video1', 'fast_video2'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label','fast_video1', 'fast_video2'])
    
    
]
val_pipeline = [
    dict(
        type='SampleFrames',
        
        
        clip_len=32,
        
        
        frame_interval=2,
        
        
        num_clips=2,
        
        
        test_mode=True),
        
        
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label', 'fast_video1', 'fast_video2'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label', 'fast_video1', 'fast_video2'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        
        
        clip_len=32,
        
        
        frame_interval=2,
        
        
        num_clips=2,
        
        
        test_mode=True),
        
        
    dict(type='FormatShape', input_format='NCTHW'),
    
    
    dict(type='Collect', keys=['imgs', 'label', 'fast_video1', 'fast_video2'], meta_keys=[]),
    
    
    dict(type='ToTensor', keys=['imgs', 'label','fast_video1', 'fast_video2'])
]


## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```



