"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import math
import cv2




class Dataset_evo_2:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    
    def __init__(self, image_size, sample_size, shape_count, r_min, r_max, noise, sigma_max):
        
        print("Initializing Random Shapes Dataset Reader...")
        self.image_size = image_size
        self.sample_size = sample_size
        self.shape_count = shape_count
        self.r_min = r_min
        self.r_max = r_max
        self.noise = noise
        self.sigma_max = sigma_max
        
        
        self._create_image_and_label()
        
    def _get_patterns(self):
        pat1 = np.array([[253,244,0],
                         [0,16,134],
                         [172,211,0]], dtype=np.uint8)
        pat2 = np.array([[211,182,0,123],
                         [199,0,221,0],
                         [0,0,0,115],
                         [251,211,0,145]], dtype=np.uint8)

        pat4 = np.array([[255,128,0,255],
                         [64,0,92,0],
                         [0,0,0,255],
                         [128,64,0,128]], dtype=np.uint8)

        pat5 = np.array([[0,0,0,0,0,0,0],
                         [0,0,164,0,0,0,0],
                         [0,0,0,0,0,0,0],
                         [0,0,0,223,0,0,0],
                         [0,0,11,0,0,0,0],
                         [0,0,211,0,0,0,0],
                         [0,0,0,0,0,0,0]], 
                        dtype=np.float64)

        pat6 = np.array([[0,0,0,0,0,0,0],
                         [0,0,64,0,0,0,0],
                         [0,0,0,0,0,0,0],
                         [0,0,0,223,0,0,0],
                         [0,123,20,90,130,0,0],
                         [0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0]], 
                        dtype=np.float64)

        #  honeycombing version 1
        pat7 = np.array([[0,0,255,0],
                         [0,255,0,255],
                         [255,0,0,0],
                         [0,255,0,255]], 
                        dtype=np.uint8)

        #  honeycombing version 2
        pat8 = np.array([[0,0,0,255,0,0],
                         [0,0,0,128,0,0],
                         [0,0,0,255,0,0],
                         [0,0,200,0,200,0],
                         [0,200,0,0,0,200],
                         [156,0,0,0,0,0],
                         [156,0,0,0,0,0],
                         [210,0,0,0,0,0],
                         [0,200,0,0,0,200],
                         [0,0,200,0,200,0]], 
                        dtype=np.uint8)

        # micronodules
        pat9 = np.array([[0,0,0,0,0,0,0],
                         [0,0,0,128,0,0,0],
                         [0,0,100,255,150,0,0],
                         [0,100,199,255,200,128,0],
                         [0,0,100,255,150,0,0],
                         [0,0,0,128,0,0,0],
                         [0,0,0,0,0,0,0]], 
                        dtype=np.uint8)

        pat10 = np.array([[0,0,0,0,0],
                          [0,0,128,0,0],
                          [0,100,255,150,0],
                          [0,0,100,0,0],
                          [0,0,0,0,0]], 
                         dtype=np.uint8)

        pat11 = np.array([[0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,255,0,0,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0]], 
                         dtype=np.uint8)
        patterns = []
        patterns.append(pat1)
        patterns.append(pat2)
        patterns.append(pat4)
        patterns.append(pat4)
        patterns.append(pat5)
        patterns.append(pat6)
        patterns.append(pat7)
        patterns.append(pat8)
        patterns.append(pat9)
        patterns.append(pat10)
        patterns.append(pat11)
        return patterns
    
    def _patch_gen(self, rows, cols, pat, pattern_selector):
        patchX = np.zeros((rows, cols), np.uint8)
        patch_label = np.zeros((rows, cols), np.uint8)

        pat_x, pat_y = pat.shape

        for i in range(math.floor(rows/pat_x)):
            for j in range(math.floor(cols/pat_y)):
                blur_factor = np.random.randint(0,20)/10        
                pat_blurred = cv2.GaussianBlur(pat,(5,5),blur_factor)
                patchX[i*pat_x:i*pat_x+pat_x,j*pat_y:j*pat_y+pat_y] = pat_blurred

                #patchX[i*pat_x:i*pat_x+pat_x,j*pat_y:j*pat_y+pat_y] = pat
                patch_label[i*pat_x:i*pat_x+pat_x,j*pat_y:j*pat_y+pat_y] = pattern_selector
        return patchX, patch_label

    def _create_image_and_label(self):
        n = self.sample_size
        nx = self.image_size
        ny = self.image_size
        cnt = self.shape_count
        noise_gen = self.noise
        noise_sigma_max = self.sigma_max
        
        pattern_list = self._get_patterns()
        
        self.images = np.zeros((n, nx, ny, 1))
        self.annotations = np.zeros((n, nx, ny, 1))

        # loop around how many records             
        for j in range(n):

            label = np.zeros((nx, ny, 1), dtype=np.int8)

            img1 = np.zeros((nx, ny), np.uint8)
            img11 = np.zeros((nx, ny), np.uint8)

            image_x_dim, image_y_dim = img1.shape

            noiselevel_selector = np.random.randint(1,noise_sigma_max)

            for i in range(cnt):

                # choose randomly the circle radius
                # radius is always in steps of the pattern used to avoid bad circle container processing
                radius_pattern_step = np.random.randint(self.r_min,self.r_max)
                pattern_selector = np.random.randint(1,11)
                
                p_x, _ = pattern_list[pattern_selector - 1].shape
                radius = p_x * radius_pattern_step
                rows = 2* radius
                cols = 2* radius
                patchX, patch_label = self._patch_gen(rows, cols, pattern_list[pattern_selector -1], pattern_selector)
                
                

                # choose randomly where to place the circle
                place_x = np.random.randint(0,image_x_dim - rows)
                place_y = np.random.randint(0,image_y_dim - cols)

                # generate the mask and its inverse of the circle container
                patch_mask = np.zeros((rows, rows), np.uint8)
                cv2.circle(patch_mask,(radius,radius), radius, (255), -1)

                mask = patch_mask
                mask_inv = cv2.bitwise_not(mask)


                # from the image  cut out the area on which we will place the new circle 
                roi = img1[ place_x : rows + place_x, place_y : cols + place_y ]

                # preserve the original image content of the container where ther is no circle
                img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

                # place the circle content in the area of the mask
                img2_fg = cv2.bitwise_and(patchX,patchX, mask = mask)

                # combine the bg and the fg 
                dst = cv2.add(img1_bg,img2_fg)

                # place the new patch into the original image
                img1[place_x:rows+place_x, place_y :cols+place_y] = dst

                if noise_gen:
                    xxx = img1.astype(float)
                    xxx += np.random.normal(scale=noiselevel_selector, size=xxx.shape)
                    xxx -= np.amin(xxx)
                    xxx /= np.amax(xxx)

                ########
                # repeat the same process for the label
                ########

                # from the image  cut out the area on which we will place the new circle 
                roi_label = img11[ place_x : rows + place_x, place_y : cols + place_y ]

                # preserve the original label content of the container where there is no circle
                img11_bg = cv2.bitwise_and(roi_label, roi_label, mask = mask_inv)

                # place the circle content in the area of the mask
                img22_fg = cv2.bitwise_and(patch_label,patch_label, mask = mask)

                 # combine the bg and the fg 
                dst_label = cv2.add(img11_bg,img22_fg)

                # place the new patch into the original image
                img11[place_x:rows+place_x, place_y :cols+place_y] = dst_label

                label = np.expand_dims(img11, axis=2)

            # no noise processing t this present time
            if noise_gen:
                
                
                self.images[j] = np.expand_dims(xxx, axis=2)
            else:
                self.images[j] = np.expand_dims(img1, axis=2)
            self.annotations[j] = label

        print (self.images.shape)
        print (self.annotations.shape)
        
        unique, counts = np.unique(self.annotations, return_counts=True)
        print("label info ------------------------------------------------ ")
        print("unique values: ", unique, "count: ", counts)
        #unique, counts = np.unique(self.images, return_counts=True)
        #print("image info ------------------------------------------------ ")
        #print("unique values: ", unique, "count: ", counts)
        

        
    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

