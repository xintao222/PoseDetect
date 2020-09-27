import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

path1 = 'C:\\Users\\74294\Desktop\imgs\\912\\*.jpg'
path2 = 'C:\\Users\\74294\Desktop\imgs\\916\\*.jpg'
path3 = 'C:\\Users\\74294\Desktop\imgs\\912-916\\'

ims1 = glob.glob(path1)
ims2 = glob.glob(path2)
ims1.sort()
ims2.sort()

for idx,im1_path in tqdm(enumerate(ims1)):
    imname = im1_path.split('\\')[-1]
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(ims2[idx])
    im = np.concatenate([im1,im2],1)
    cv2.imwrite(os.path.join(path3, imname),im)
