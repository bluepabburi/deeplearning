# import sys, os

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# from dataset.mnist import load_mnist

# (x_train, t_train),(x_test, t_test) = \
#     load_mnist(flatten= True, normalize=False)

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img= Image.fromarray(np.uint8(img))
    pil_img.show()



(x_train, t_train),(x_test, t_test) = \
    load_mnist(flatten= True, normalize=False)


img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)