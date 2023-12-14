_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/aphid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# FULL Weights: [1.0888087035747884, 12.260157616847424]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0,
            class_weight=[1.0888087035747884, 12.260157616847424]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0,
            class_weight=[1.0888087035747884, 12.260157616847424])],
        num_classes=2),
    auxiliary_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0,
            class_weight=[1.0888087035747884, 12.260157616847424]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0,
            class_weight=[1.0888087035747884, 12.260157616847424])],
        num_classes=2))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
evaluation = dict(interval=1000, metric=['mIoU', 'mFscore', 'mDice'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=1000)