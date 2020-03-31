from metrics import calc_metrics
import metrics
import glob
import numpy as np
import os
import cv2
import kitti_util

dense_dir = '/mnt/dbserver01-mtl-srv2/fm/stereo/GANet/ax_seq_01/Kitti2015'
lidar_dir = '/media/frank.julca-aguilar/daimler_data/gated_sequences/Algolux_sequences_v1/lidar_hdl64_last_stereo_left'
calib_path = '/data0/frank/dev/datasets/tracking/kitti/3d_obj_detection/data_object_image_2/training/calib/calib_stereo_kitti_last_tobi.txt'

def disparity_to_distance(calib, disp, is_kitti):
    disp[disp < 0] = 0
    if is_kitti:
        baseline = 0.54 # 0.203 # 0.54
    else:
        baseline = 0.203 #0.203
    
    print('baseline: ', baseline)
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    return depth
    

def read_map(fname, is_lidar=False, target_shape=None):
    if is_lidar:
        return np.load(fname)['arr_0']
    else:
        disp_map = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        disp_map = cv2.resize(disp_map, target_shape, interpolation=cv2.INTER_AREA)

        #TODO: check difference in metrics when removing padded regions
        disp_scale = target_shape[0] / float(disp_map.shape[1])
        disp_map = disp_map * disp_scale

        return disp_map

accumulated = []
calib = kitti_util.calib(calib_path)
for fn in glob.glob(os.path.join(dense_dir, '*.png'))[:10]:
    lidar_map = read_map(os.path.join(lidar_dir, os.path.basename(fn)[:-14] + '.npz'), is_lidar=True)
    dense_map = read_map(fn, is_lidar=False, target_shape=(lidar_map.shape[1], lidar_map.shape[0]))
    dense_map = disparity_to_distance(calib, dense_map, is_kitti=False)   
    
    results = calc_metrics(dense_map, lidar_map, min_distance=3., max_distance=150.)
    accumulated.append(results)
    print(results)

print(metrics.metric_str)
print('mean: ', np.mean(accumulated, axis=0, dtype=np.float64))
