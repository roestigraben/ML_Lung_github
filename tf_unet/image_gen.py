# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import cv2
import math
from tf_unet.image_util import BaseDataProvider

class KaggleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(KaggleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)

class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)



class RgbDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(RgbDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

        
    def _next_data(self):
        data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
        return to_rgb(data), label

class DataProvider(BaseDataProvider):
    channels = 1
    n_class = 3
    
    def __init__(self, nx, ny, **kwargs):
        super(DataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        self.n_class=3
        
        
    def _next_data(self):
        return create_image_and_label_with_patterns(self.nx, self.ny, **self.kwargs)

def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 0.1, rectangles=False):
    
    
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r

        mask = np.logical_or(mask, m)

        image[m] = h

    label[mask, 1] = 1
    
    if rectangles:
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt//2):
            a = np.random.randint(nx)
            b = np.random.randint(ny)
            r =  np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)
    
            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h
            
        label[mask, 2] = 1
        
        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    if rectangles:
        # shape should be (512, 512, 1) (512, 512, 3)
        print(image.shape, label.shape)
        print(np.unique(image))
        print(np.unique(label))
        return image, label
    else:
        return image, label[..., 1]

def create_image_and_label_with_patterns(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
    
    pat1 = np.array([[1,1,0],[1,0,1],[1,1,0]], dtype=np.uint8)
    pat2 = np.array([[1,1,0,1],[1,0,1,0],[0,0,0,1],[1,1,0,1]], dtype=np.uint8)

    pat4 = np.array([[255,128,0,255],[64,0,92,0],[0,0,0,255],[128,64,0,128]], dtype=np.uint8)

    _, pat5 = cv2.threshold(pat1, 0.5, 255, cv2.THRESH_BINARY)
    _, pat6 = cv2.threshold(pat2, 0.5, 255, cv2.THRESH_BINARY)

    label = np.zeros((nx, ny, 3), dtype=np.uint8)
    image = np.zeros((nx, ny), np.uint8)

    image_x_dim, image_y_dim = image.shape

    for i in range(cnt):

        radius_pattern_step = np.random.randint(1,10)
        pattern_selector = np.random.randint(1,3)
        
        if pattern_selector == 1:
            p_x, _ = pat5.shape
            radius = p_x * radius_pattern_step
            rows = 2* radius
            cols = 2* radius
            patchX, patch_label = patch_gen(rows, cols, pat5, pattern_selector)
        elif pattern_selector == 2:
            p_x, _ = pat4.shape
            radius = p_x * radius_pattern_step
            rows = 2* radius
            cols = 2* radius
            patchX, patch_label = patch_gen(rows, cols, pat4, pattern_selector)
        elif pattern_selector == 3: 
            p_x, _ = pat6.shape
            radius = p_x * radius_pattern_step 
            rows = 2* radius
            cols = 2* radius
            patchX, patch_label = patch_gen(rows, cols, pat6, pattern_selector)

        # choose randomly where to place the circle
        place_x = np.random.randint(0,image_x_dim - rows)
        place_y = np.random.randint(0,image_y_dim - cols)

        # generate the mask and its inverse of the circle container
        patch_mask = np.zeros((rows, rows), np.uint8)
        cv2.circle(patch_mask,(radius,radius), radius, (255), -1)

        mask = patch_mask
        mask_inv = cv2.bitwise_not(mask)

        # from the image  cut out the area on which we will place the new circle 
        roi = image[ place_x : rows + place_x, place_y : cols + place_y ]

        # preserve the original image content of the container where ther is no circle
        img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

        # place the circle content in the area of the mask
        img2_fg = cv2.bitwise_and(patchX, patchX, mask = mask)

        # combine the bg and the fg 
        dst = cv2.add(img1_bg,img2_fg)

        # place the new patch into the original image
        image[place_x:rows+place_x, place_y :cols+place_y] = dst

        ########
        # repeat the same process for the label
        ########
        
        # from the image  cut out the area on which we will place the new circle 
        roi_label = label[ place_x : rows + place_x, place_y : cols + place_y , pattern_selector]
        roi_label_bg = label[ place_x : rows + place_x, place_y : cols + place_y , 0]

        # preserve the original label content of the container where there is no circle
        img11_bg = cv2.bitwise_and(roi_label, roi_label, mask = mask_inv)
        img11_bg_bg = cv2.bitwise_and(roi_label_bg, roi_label_bg, mask = mask_inv)

        # place the circle content in the area of the mask
        img22_fg = cv2.bitwise_and(patch_label, patch_label, mask = mask)
        img22_fg_bg = cv2.bitwise_and(patch_label, patch_label, mask = mask)
        
         # combine the bg and the fg 
        dst_label = cv2.add(img11_bg, img22_fg)
        dst_label_bg = cv2.add(img11_bg_bg, img22_fg_bg)
        
        # place the new patch into the original image
        label[place_x:rows+place_x, place_y :cols+place_y, pattern_selector] = dst_label
        label[place_x:rows+place_x, place_y :cols+place_y, 0] = dst_label_bg

    # invert mask
    label[...,0] = label[...,0]*-1+1

    image = np.expand_dims(image, axis=2)

    image = image.astype(np.float64)

    image = image / 255.
    image += np.random.normal(scale=0, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    #image = np.expand_dims(image, axis=2)

    label = label.astype(np.bool)

    
    return image, label
   

def patch_gen(rows, cols, pat, pattern_selector):
    #print("rows : ", rows, "Columns: ", cols)
    patchX = np.zeros((rows, cols), np.uint8)
    patch_label = np.zeros((rows, cols), np.uint8)
    
    pat_x, pat_y = pat.shape

    for i in range(math.floor(rows/pat_x)):
        for j in range(math.floor(cols/pat_y)):
            patchX[i*pat_x:i*pat_x+pat_x,j*pat_y:j*pat_y+pat_y] = pat
            patch_label[i*pat_x:i*pat_x+pat_x,j*pat_y:j*pat_y+pat_y] = 1
    return patchX, patch_label

def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

