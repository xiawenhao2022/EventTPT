import os
import os.path
import torch
import numpy as np
import pandas
import csv
from glob import glob
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame
import cv2


class VisEvent(BaseVideoDataset):
    """ VisEvent dataset.
    """

    def __init__(self, root=None, dtype='rgbrgb', split='train', image_loader=jpeg4py_loader_w_failsafe): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        root = env_settings().visevent_dir if root is None else root
        assert split in ['train'], 'Only support train split in VisEvent, got {}'.format(split)
        super().__init__('VisEvent', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._build_sequence_list()
        self.detection_boxes = [{} for _ in range(5)]

    def _apply_contrast(self, pixels, weight):
        # 应用对比度处理，通过修改像素值的权重
        return cv2.convertScaleAbs(pixels, alpha=weight, beta=0)
    def _update_detection_boxes(self, new_box):
        # 将新的检测框信息添加到列表的开头，并移除最后一个元素以保持列表长度为5
        self.detection_boxes.insert(0, new_box)
        if len(self.detection_boxes) > 5:
            self.detection_boxes.pop()

    def _get_frame_with_detections(self, seq_path, frame_id, anno):
        
        
        # 获取当前帧的RGB和事件图像
        color_path, event_path = self._get_frame_path(seq_path, frame_id)
        event_img = get_x_frame(event_path, None, dtype='color', depth_clip=False)  # 假设'event'处理事件图像
        rgb_img = get_x_frame(color_path, None, dtype='color', depth_clip=False)  # 获取RGB图像
        import pdb;pdb.set_trace()
        
        # 根据anno信息获取当前帧的GT检测框的坐标和尺寸
        bbox = anno['bbox'][frame_id]
        #import pdb;pdb.set_trace()
        y1, x1, y2, x2 = bbox[:4].int()
        if y1 < 0 or y2 > rgb_img.shape[0] or x1 < 0 or x2 > rgb_img.shape[1]:
            raise ValueError("Invalid bounding box coordinates")
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        # 从RGB图像中裁剪出GT检测框内的像素
        current_frame_pixels = event_img[y2:y1, x2:x1]

        #print('############################')
        
        # 更新检测框列表
        self.detection_boxes.append({
            'coordinates': [x1, y1, x2, y2],
            'pixels': current_frame_pixels
        })
        if len(self.detection_boxes) > 5:
            self.detection_boxes.pop(0)  # 保持列表长度为5

        #print('############################')    
        
        
        
        # 将列表中的检测框像素粘贴到事件图像上，并应用不同的权重
        for i, box in enumerate(reversed(self.detection_boxes)):
            if box['pixels'] is not None:
                # 计算权重，权重随列表位置增加而减少
                weight = 0.2 + 0.1 * i  # 从0.2开始递增，步长为0.1
                # 调整叠加图权重，使得权重越小的图像块越淡
                for c in range(len(box['pixels'])):
                    #print('############################')  
                    box['pixels'][:, :, c] = cv2.convertScaleAbs(box['pixels'][:, :, c], alpha=weight, beta=0)
                #print('############################')  
                
                # 将调整过权重的像素块粘贴到事件图像上
                event_img[y2:y1, x2:x1] = box['pixels']
                #import pdb;pdb.set_trace()
        #print('############################')   
        #import pdb;pdb.set_trace()
        # 进行RGB和事件图像的融合处理
        final_img = cv2.addWeighted(rgb_img, 1, event_img, 0.5, 0)
        #import pdb;pdb.set_trace()
        
        
        return final_img

    def _build_sequence_list(self):

        file_path = os.path.join(self.root, '{}list.txt'.format(self.split))
        sequence_list = pandas.read_csv(file_path, header=None).values.tolist()
        return sequence_list

    def get_name(self):
        return 'visevent'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absent_label.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in list(csv.reader(f))])

        target_visible = occlusion

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        print(f"seq_id: {seq_id}, seq_name: {seq_name}, type: {type(seq_name)}")
        # 确保 seq_name 是字符串，如果是列表，使用 ''.join(seq_name) 将其转换为字符串
        if isinstance(seq_name, list):
            seq_name = '/'.join(seq_name)
        # 现在 seq_name 应该是一个字符串，可以安全地用于 os.path.join
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 5.0) & (bbox[:, 3] > 5.0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return rgb event image path
        '''
        vis_img_files = sorted(glob(os.path.join(seq_path, 'vis_imgs', '*.bmp')))

        try:
            vis_path = vis_img_files[frame_id]
        except:
            print(f"seq_path: {seq_path}")
            print(f"vis_img_files: {vis_img_files}")
            print(f"frame_id: {frame_id}")

        event_path = vis_path.replace('vis_imgs', 'event_imgs')


        return vis_path, event_path # frames start irregularly

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - rgb+event_colormap
        '''
        color_path, event_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(color_path, event_path, dtype=self.dtype, depth_clip=False)
        return img  # (h,w,6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        self.detection_boxes = [{} for _ in range(5)]
        seq_path = self._get_sequence_path(seq_id)
        
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]
        
        
        frame_list = [self._get_frame_with_detections(seq_path, f_id, anno) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

