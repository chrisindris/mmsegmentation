_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/aphid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    decode_head=dict(
        type='FCNHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=sum([18, 36, 72, 144]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0,
            class_weight=[1.0888087035747884, 12.260157616847424]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0,
            class_weight=[1.0888087035747884, 12.260157616847424])
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
evaluation = dict(interval=1000, metric=['mIoU', 'mFscore', 'mDice'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=1000)
