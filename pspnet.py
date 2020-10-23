import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50

# pool with different bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):
    def __init__(self,bin_sizes=[1,2,3,6]):
        super(PSPModule,self).__init__()
        self.bin_sizes = bin_sizes
        self.features = []
        self.conv = []
        c = 2048
        for size in bin_sizes:
            self.conv.append(Conv2D(c,c//4,size))
        
    def forward(self,inputs):
        self.features = []
        for idx,bin_size in enumerate(self.bin_sizes):
            self.features.append(fluid.layers.adaptive_pool2d(inputs,bin_size))
            x =  self.features[idx]
            x= self.conv[idx](x)
            x = fluid.layers.interpolate(x,out_shape=inputs.shape[2:],align_corners=False)
            self.features[idx] = x
        self.features.append(inputs)
        x = fluid.layers.concat(input=self.features,axis=1)
        return x


class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()
        res = ResNet50()
        
        #self.backbone = res
        # stem: res.conv, res.pool2d_max
        self.layer0 = fluid.dygraph.Sequential(*[res.conv,res.pool2d_max])
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        # psp: 2048 -> 2048*2
        self.psp = PSPModule()

        # cls: 2048*2 -> 512 -> num_classes
        self.conv1 = Conv2D(2048*2,512,3,padding=1)
        self.conv2 = Conv2D(512,num_classes,1)

        # aux: 1024 -> 256 -> num_classes
        
    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.psp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = fluid.layers.interpolate(x,inputs.shape[2:],align_corners=True)
        # aux: tmp_x = layer3
        return x
            



def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data=np.random.rand(2,3, 473, 473).astype(np.float32)
        print(x_data.shape)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        model.train()
        pred, aux = model(x)
        print(pred.shape)#, aux.shape)

if __name__ =="__main__":
    main()
