# Dataset Settings
dataset_type = 'AphidDataset'
data_root = 'data/FILTERED'
img_norm_cfg = dict(
    mean=[0.2377, 0.3481, 0.3058], std=[0.2299, 0.2375, 0.2250], to_rgb=True)   # FILTERED
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 768), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DetSegImages',
        ann_dir='05-05',
        split='ImageSets/Main/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DetSegImages',
        ann_dir='05-05',
        split='ImageSets/Main/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DetSegImages',
        ann_dir='05-05',
        split='ImageSets/Main/val.txt',
        pipeline=test_pipeline))
