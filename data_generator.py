import glob
import numpy as np
import cv2

patch_size, stride = 40, 32
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


#数据增强
def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):
    #read multispectral image
    img = np.load(file_name)
    h,w,s = img.shape
    patches = []
    for scale in scales:
        h_scaled,w_scaled = int(h*scale),int(w*scale)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        #提取块
        for i in range(0,h_scaled-patch_size+1,stride):
            for j in range(0,w_scaled-patch_size+1,stride):
                x = img_scaled[i:i+patch_size,j:j+patch_size,:]
                for k in range(0,aug_times):
                    x_aug = data_aug(x,mode=np.random.randint(0,8))
                    patches.append(x_aug)

    return patches




def datagenerator(data_dir='data/MSI_data_npy',verbose=False):
    file_list = glob.glob(data_dir + '/*.npy')
    cv2.imshow("",file_list[1][:,:,25])
    data = []
    for i in range (len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i+1)+'/'+str(len(file_list))+' is done')
    data = np.array(data,dtype='uint8')
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4],1))
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis=0)
    print('training data finished')
    return data

if __name__ == '__main__':
    data = datagenerator(data_dir='data/MSI_data_npy')
