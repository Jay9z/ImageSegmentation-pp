import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid

class Transform():
    def __init__(self,size=256):
        self.size=size
    
    def __call__(self,image,label):
        img = cv2.resize(image,(self.size,self.size),interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        return img,label

class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        self.paths = self.read_list()

    def read_list(self):
        fp = open(self.image_list_file)
        path_list = []
        for line in fp.readlines():
            paths = line.split()
            image_path,lab_path = paths[0],paths[1]
            image_path = os.path.join(self.image_folder,image_path)
            lab_path = os.path.join(self.image_folder,lab_path)
            path_list.append((image_path,lab_path))
        if self.shuffle:
            random.shuffle(path_list)

        return path_list

    def preprocess(self, data, label):
        h_1,w_1,c_1 = data.shape
        h_2,w_2 = label.shape
        assert h_1==h_2, "no equal height"
        assert w_1==w_2, "no equal width"

        if self.transform:
            data,label = self.transform(data,label)
        label = label[:,:,np.newaxis]
        return data,label

    def __len__(self):
        return len(self.paths)

    def __call__(self):
        for image_path,lab_path in self.paths:
            data = cv2.imread(image_path,cv2.IMREAD_COLOR)
            label = cv2.imread(lab_path,cv2.IMREAD_GRAYSCALE)
            yield self.preprocess(data,label)


def main():
    batch_size = 5
    #place = fluid.CPUPlace()
    transform = Transform()
    #place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard():
        # TODO: craete BasicDataloder instance
        image_folder="work/dummy_data"
        image_list_file="work/dummy_data/list.txt"
        data = BasicDataLoader(image_folder,image_list_file,transform=transform)

        # TODO: craete fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        # TODO: set sample generator for fluid dataloader 
        dataloader.set_sample_generator(data,batch_size)

        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')

if __name__ == "__main__":
    main()
