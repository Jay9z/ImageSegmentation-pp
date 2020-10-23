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


args = parser.parse_args()


def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        #TODO:
        images, labels = data

        images = fluid.layers.transpose(images,(0,3,1,2))

        preds = model(images)
        loss = criterion(preds,labels)

        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = images.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg


def main():
    # Step 0: preparation
    #place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard():
        # Step 1: Define training dataloader
        image_folder="work/dummy_data"
        image_list_file="work/dummy_data/list.txt"
        transform = TrainAugmentation(224)
        data = BasicDataLoader(image_folder,image_list_file,transform=transform)
        #TODO: create dataloader
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=2,return_list=True)
        train_dataloader.set_sample_generator(data,args.batch_size)
        total_batch = len(data)//args.batch_size
        # Step 2: Create model
        if args.net == "basic":
            #TODO: create basicmodel
            model = PSPNet()
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

        # Step 3: Define criterion and optimizer
        criterion = Basic_SegLoss

        # create optimizer
        optimizer = AdamOptimizer(learning_rate=args.lr,parameter_list=model.parameters())
        # Step 4: Training
        for epoch in range(1, args.num_epochs+1):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}")

                # TODO: save model and optmizer states
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict,model_path)
                optim_dict = optimizer.state_dict()
                fluid.save_dygraph(optim_dict,model_path)

                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')


if __name__ == "__main__":
    main()
