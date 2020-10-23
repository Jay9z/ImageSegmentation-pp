
from PIL import Image
import cv2
from pspnet import PSPNet
import os
from basic_dataloader import BasicDataLoader
from basic_data_preprocessing import *
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
import numpy as np
import pdb
from basic_transforms import *
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from pspnet import PSPNet
from basic_dataloader import BasicDataLoader
from basic_seg_loss import Basic_SegLoss
from basic_data_preprocessing import TrainAugmentation


parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--inf_type', type=int, default=0)


args = parser.parse_args()


def colorize(gray, palette=2):
    '''
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    '''
    # gray: numpy array of the label and 1*3N size list palette
    # color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    # color.save("color_input.png")
    # data = np.array(color)
    # print(np.min(data),np.max(data),data.shape)
    # #color.putpalette(palette)
    gray = gray.astype(np.uint8)
    color = cv2.applyColorMap(gray,2)
    return color

def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)


def inference_resize(model,images):
    tm = Compose([
        Resize(224)  
        ]
        )

    return tm(images)[0]

def inference_sliding():
    pass

def inference_multi_scale():
    pass


# def save_images(images,suffix="input"):
#     ## images
#     # print(images.shape,type(images))
#     if isinstance(images,Image.Image):
#         images = np.array(images)
#     assert isinstance(images,np.ndarray), "wrong image data type" 
#     print(images.shape)
#     n = images.shape[0]
#     for i in range(n):
#         #image = np.transpose(images[i],(1,2,0))
#         image = images[i]
#         image = Image.fromarray(image.astype(np.uint8))#.convert('P')
#         image.save(f"{i}_{suffix}.png")


# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    with fluid.dygraph.guard():
        # 1. create model
        model = PSPNet(num_classes=59)

        # 2. load pretrained model 
        pretrain_file = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{args.num_epochs}")
        if os.path.exists(pretrain_file):
            state,_ = fluid.load_dygraph(pretrain_file)
            model.set_dict(state)

        # 3. read test image list
        image_folder = "work/dummy_data"
        image_list_file = "work/dummy_data/list.txt"
        transform = None
        if args.inf_type == 0:
            transform = Augmentation0()
        elif args.inf_type == 1:
            transform = Augmentation1()
        elif args.inf_type == 2:
            transform = Augmentation2()
        elif args.inf_type == 3:
            transform = Augmentation3()
        elif args.inf_type == 4:
            transform = Augmentation4()
        data = BasicDataLoader(image_folder,image_list_file,transform=transform)
        # TODO: craete fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        # TODO: set sample generator for fluid dataloader 
        dataloader.set_sample_generator(data,1)

        # 4. create transforms for test image, transform should be same as training
        

        # 5. loop over list of images
        for idx,data in enumerate(dataloader):
            images,labels = data
            #images = transform(images)
            # 6. read image and do preprocessing
            #print(images.shape)
            i_file = f"{idx}_input.png"
            input_images = (images*255.0).numpy().astype(np.uint8)
            #print(np.max(input_images[0]),np.min(input_images[0]),input_images.shape)
            image =input_images[0].astype(np.uint8)
            cv2.imwrite(i_file,image)

            images = fluid.layers.transpose(images,(0,3,1,2))
            
            # 7. image to variable
            images = to_variable(images)

            # 8. call inference func
            preds = model(images).numpy()
            pred = np.squeeze(preds)
            result = np.argmax(pred,axis=0)
            
            #print(preds.shape,result.shape)

            # 9. save results
            result = colorize(result,[1,2,3])
            o_file = f"{idx}_pred.png"
            cv2.imwrite(o_file,result)
            
            save_blend_image(i_file,o_file)


if __name__ == "__main__":
    main()
