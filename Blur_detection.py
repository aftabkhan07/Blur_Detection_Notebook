import numpy as np
from skimage.filters import laplace, sobel, roberts
import pickle
import cv2
import os


def get_one_data(path,img):
    feature=[]
    image_gray = cv2.imread(path+img,0)
    lap_feat = laplace(image_gray)
    sob_feat = sobel(image_gray)
    rob_feat = roberts(image_gray)
    feature.extend([img,lap_feat.mean(),lap_feat.var(),np.amax(lap_feat),
                    sob_feat.mean(),sob_feat.var(),np.max(sob_feat),
                    rob_feat.mean(),rob_feat.var(),np.max(rob_feat)])
    return feature

pickled_model = pickle.load(open('blurNotBlur.pkl', 'rb'))

_PATH = "./input/Download_folder/"
for img_name in os.listdir(_PATH):
    test_image1 = get_one_data(_PATH,img_name)
    test_image1 = np.array([test_image1[1:]])
    pred = pickled_model.predict(test_image1)

    if round(pred[0]) == 1:
        print (f"{img_name} image is clear")
    else:
        print (f"{img_name} Image is blurry")


