class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/pretrained_networks'
        self.got10k_val_dir = '/data/got10k/val'
        self.lasot_lmdb_dir = '/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/coco_lmdb'
        self.coco_dir = '/data/coco'
        self.lasot_dir = '/data/lasot'
        self.got10k_dir = '/data/got10k/train'
        self.trackingnet_dir ='/data/trackingnet'
        self.depthtrack_dir = '/data/depthtrack/train'
        self.lasher_dir = '/data/lasher/trainingset'
        self.visevent_dir = 'data/VisEvent/train'
