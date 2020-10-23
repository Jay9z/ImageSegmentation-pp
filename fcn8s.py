import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
 # TODO: create fcn8s model
    def __init__(self,num_classes=59):
        super(FCN8s,self).__init__()
        vgg16bn = VGG16BN()
        self.layer1 = vgg16bn.layer1
        self.layer1[0].conv._padding = [100,100]
        self.layer2 = vgg16bn.layer2
        self.layer3 = vgg16bn.layer3
        self.layer4 = vgg16bn.layer4
        self.layer5 = vgg16bn.layer5

        # self.conv1_1 = Conv2D(3,64,3,padding=1)
        # self.conv1_2 = Conv2D(64,64,3,padding=1)
        self.pool1 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        # self.conv2_1 = Conv2D(64,128,3,padding=1)
        # self.conv2_2 = Conv2D(128,128,3,padding=1)
        self.pool2 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        # self.conv3_1 = Conv2D(128,256,3,padding=1)
        # self.conv3_2 = Conv2D(256,256,3,padding=1)
        # self.conv3_3 = Conv2D(256,256,3,padding=1)
        self.pool3 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        # self.conv4_1 = Conv2D(256,512,3,padding=1)
        # self.conv4_2 = Conv2D(512,512,3,padding=1)
        # self.conv4_3 = Conv2D(512,512,3,padding=1)
        self.pool4 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        # self.conv5_1 = Conv2D(512,512,3,padding=1)
        # self.conv5_2 = Conv2D(512,512,3,padding=1)
        # self.conv5_3 = Conv2D(512,512,3,padding=1)
        self.pool5 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        self.conv6 = Conv2D(512,4096,1,act='relu')
        self.conv7 = Conv2D(4096,4096,1,act='relu')
        self.drop6 = Dropout()
        self.drop7 = Dropout()

        self.score = Conv2D(4096,num_classes,1)
        self.score_pool3 = Conv2D(256,num_classes,1)
        self.score_pool4 = Conv2D(512,num_classes,1,)
        self.upsample1 = Conv2DTranspose(num_classes,num_classes,filter_size=4,stride=2,padding=2,bias_attr=False)
        self.upsample2 = Conv2DTranspose(num_classes,num_classes,filter_size=4,stride=2,padding=2,bias_attr=False)
        self.upsample3 = Conv2DTranspose(num_classes,num_classes,filter_size=16,stride=8,padding=1,bias_attr=False)

    def forward(self,inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        pool3 = x
        x = self.layer4(x)
        x = self.pool4(x)
        pool4 = x
        x = self.layer5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.drop6(x)
        x = self.conv7(x)
        x = self.drop7(x)
        print(x.numpy().shape)
        x = self.score(x) # 1/32
        print(x.numpy().shape)
        x = self.upsample1(x)
        print(x.numpy().shape)
        output_final = x  # 1/16

        x = self.score_pool4(pool4)
        x = x[:,:,0:0+output_final.shape[2],0:0+output_final.shape[3]]

        up_pool4 = x
        x = up_pool4+ output_final
        x = self.upsample2(x) 
        output_4 = x # 1/8
        x = self.score_pool3(pool3)
        up_pool3 = x[:,:,0:0+output_4.shape[2],0:0+output_4.shape[3]]
        x = up_pool3+ output_4
        x = self.upsample3(x) 
        x = x[:,:,0:0+inputs.shape[2],0:0+inputs.shape[3]] #1/1
        return x
   

def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)


if __name__ == '__main__':
    main()
