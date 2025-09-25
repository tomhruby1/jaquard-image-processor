import cv2
import numpy as np
from pathlib import Path

# modra = pozadi
# cervena = lic
# zelena = rub
# zluta = oboji
BLUE   = (0,0,255)
RED    = (255,0,0) 
GREEN  = (0,255,0)
YELLOW = (255,255,0)



# jacquard 3 colors: TODO: define this somewhere as a user
# indices here map it to the row indices in the output
color_order = ((  1, 194,  83),
               ( 12,  88,  17),
               (255, 142, 246))


def img2jacquard(front_img:np.ndarray, back_img:np.ndarray, color_order:tuple)->np.ndarray:
    '''
    Make n x m x 3 image patern into n*3 x m (x 3) jacquard pattern
    args:
        - front_image: lic
        - back_image: rub of the same shape as front_image
    '''
    assert len(color_order) == 3
    assert front_img.shape == back_img.shape

    res = np.zeros((front_img.shape[0]*3, front_img.shape[1], 3))

    for i in range(res.shape[0]):
        if i % 3 == 0:
            for j in range(res.shape[1]):
                # filling in row i -> row i+3
                res[i:i+3,j,:] = BLUE # default blue
                if np.all(front_img[i//3, j, :] == back_img[i//3, j, :]):
                    res[i+color_order.index(tuple(front_img[i//3, j, :])), j, :] = YELLOW
                else:
                    res[i+color_order.index(tuple(front_img[i//3, j, :])), j, :] = RED
                    res[i+color_order.index(tuple(back_img[i//3, j, :])), j, :] = GREEN

    res = res.astype(np.uint8)

    return res

def read_image(img_path:Path)->np.ndarray:
    img = cv2.imread("data/celasala_3.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def generate_noise_like(img):
    ''' generate random 3-channel noise using ref. image'''
    color_set = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    random_indices = np.random.choice(3, size=img.shape[:-1])
    noise_img = color_set[random_indices]

    return noise_img