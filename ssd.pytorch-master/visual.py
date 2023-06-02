import torch.optim as optim
from options.train_options import TrainOptions
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from models import networks
from options.test_options import TestOptions
from util import util
import time



def test():
    opt = TrainOptions().parse()  # get testing options

    model = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                              not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

    print('Loading……………………………………')
    state_dict = torch.load('./weights/modelA_5500.pth')
    model.load_state_dict(state_dict)
    model.cuda()

    # set up the image transformation pipeline
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # set up the input and output folders
    input_folder = '../datasets/uiebd/testA'
    output_folder = './finetune_results'

    # create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # loop through all the images in the input folder
    for file_name in os.listdir(input_folder):
        print(file_name)
        # load the input image
        input_image = Image.open(os.path.join(input_folder, file_name))

        # apply the transformation pipeline
        input_tensor = transform(input_image)
        input_tensor = input_tensor.unsqueeze(0).cuda()

        # run the model and get the output
        with torch.no_grad():

            # torch.cuda.synchronize()
            # start = time.time()
            output_tensor = model(input_tensor)

            # torch.cuda.synchronize()
            # end = time.time()
            # p_time = end - start
            # print("================================= time for %f============================" % (p_time))

        # convert the output tensor to an image
        im = util.tensor2im(output_tensor)
        output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

        # save the output image
        output_file_name = os.path.splitext(file_name)[0] + '.png'
        output_path = os.path.join(output_folder, output_file_name)
        util.save_image(im, output_path)

if __name__ == '__main__':
    # train()
    test()