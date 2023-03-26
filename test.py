import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
import torchvision


def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255
    img = K.color.bgr_to_rgb(img)
    return img


fname1 = '1.jpg'
fname2 = '2.jpg'

img1 = load_torch_image(fname1)
img2 = load_torch_image(fname2)

matcher = KF.LoFTR(pretrained='outdoor')

input_dict = {'image0': K.color.rgb_to_grayscale(img1),
              'image1': K.color.rgb_to_grayscale(img2)}

with torch.no_grad():
    correspondences = matcher(input_dict)

for k, v in correspondences.items():
    print(k)

mask = correspondences['confidence'] > 0.9
indices = torch.nonzero(mask)
correspondences['confidence'] = correspondences['confidence'][indices]
correspondences['keypoints0'] = correspondences['keypoints0'][indices]
correspondences['keypoints1'] = correspondences['keypoints1'][indices]
correspondences['batch_indexes'] = correspondences['batch_indexes'][indices]

mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()
H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

# Функция отрисовки совпадений
draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                 torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={'inlier_color': (0.2, 1, 0.2),
               'tentative_color': None,
               'feature_color': (0.2, 0.2, 1), 'vertical': False})

plt.axis('off')
plt.savefig('matches1.png')
plt.show()

