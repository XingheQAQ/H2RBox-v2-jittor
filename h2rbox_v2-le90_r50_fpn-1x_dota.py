# model settings
model = dict(
    type='H2RBoxV2',
    crop_size=(1024, 1024),
    view_range=(0.25, 0.75),
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        norm_eval=True,
        return_stages=["layer1", "layer2", "layer3", "layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
        relu_before_extra_convs=True),
    roi_heads=dict(
        type='H2RBoxV2Head',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        scale_angle=False,
        rotation_agnostic_classes=[1, 9, 11],
        agnostic_resize_classes=[1],
        use_circumiou_loss=True,
        use_standalone_angle=True,
        use_reweighted_loss_bbox=False,
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_bce=True, loss_weight=1.0),
        loss_symmetry_ss=dict(
            type='H2RBoxV2Loss',
            use_snap_loss=True,
            loss_rot=dict(type='SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp=dict(type='SmoothL1Loss', loss_weight=0.05, beta=0.1)),
        test_cfg=dict(
            centerness_factor=0.5,
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='obb_nms', iou_thr=0.1),
            max_per_img=2000)
    ))

# dataset settings
dataset = dict(
    train=dict(
        type="DOTAWSOODDataset",
        dataset_dir='autodl-tmp/data/DOTA1.1/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                 min_size=1024,
                 max_size=1024
            ),
            dict(
                type='RotatedRandomFlip',
                prob=0.75
            ),
            dict(
                type="Pad",
                size_divisor=32
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False)
        ],
        batch_size=1,
        num_workers=2,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="DOTAWSOODDataset",
        dataset_dir='autodl-tmp/data/DOTA1.1/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize", 
                min_size=1024,
                max_size=1024
            ),
            dict(
                type="Pad", 
                size_divisor=32
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=4,
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir='autodl-tmp/data/DOTA1.1/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize", 
                min_size=1024, 
                max_size=1024
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        num_workers=2,
        batch_size=4,
    )
)

# optimizer settings
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

# scheduler settings
scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[8, 11]
)


# logger settings
logger = dict(
    type="RunLogger"
)

# training settings
max_epoch = 12
eval_interval = 6
checkpoint_interval = 1
log_interval = 50
