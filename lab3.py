from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

imsize = 512
# imdim = (1280,720)

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_w = {'conv_1': 0.75,
      'conv_2': 0.5,
      'conv_3': 0.2,
      'conv_4': 0.2,
      'conv_5': 0.2}


def image_loader(image_name,image_dim=None):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    if (image_dim):
      image = image.resize(image_dim)
    else:
      image = image.resize((imsize,imsize))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# style_img = image_loader("circle1.jpg")
# content_img = image_loader("lastsupper.jpg")

# style_img = image_loader("circle1.jpg")
# content_img = image_loader("lastsupper.jpg")


# assert style_img.size() == content_img.size(), \
#     "we need to import style and content images of the same size"


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# plt.figure()
# imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature,sw):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        # print(sw)
        self.sw = sw

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.sw * F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,style_weights=style_w):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature,style_w[name])
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
# plt.figure()
# imshow(input_img, title='Input Image')

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()                  
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, input_img,num_steps=500)

# plt.figure()
# imshow(output, title='Output Image')

# output_img = output.squeeze(0)      # remove the fake batch dimension
# output_img = unloader(output_img)
# output_img.save("output_lastsupper_circles1.jpg")

def main():
    
    def evaluate_content_path(arg):
        if (os.path.exists(arg)):
            return arg
        else:
            raise argparse.ArgumentTypeError('The content path does not exist\n')


    def evaluate_style_path(arg):
        if (os.path.exists(arg)):
            return arg
        else:
            raise argparse.ArgumentTypeError('The style path does not exist\n')

    def evaluate_epochs_arg(arg):
        if (int(arg) > 0):
            return arg
        else:
            raise argparse.ArgumentTypeError("The number of epochs specified was less than or equal to 0.\n")

    parser = argparse.ArgumentParser(description='Options to run lab3')
    parser.add_argument('-s','--style_image_path', type=evaluate_style_path,
                    help='Path to style image\n')
    
    parser.add_argument('-c','--content_image_path', type=evaluate_content_path,
                    help="Path to content image\n")
    
    parser.add_argument('-e','--epochs',type=evaluate_epochs_arg,help="Number of epochs. Must be greater than 0.")

    parser.add_argument('-v','--view_before_saving', type=bool,
                    help="True or False. True for displaying styled transfer image after every 50 epochs and having the option to save. (Note: epoch argument will be ignored). False for saving after the specified number of epochs.\n")
    
    args = parser.parse_args()

    # if not (any(vars(args).values())):
    #     parser.error('One or more of these arguments were not supplied: -s or --style_image_path and -c or --content_image_path and -e or --epochs and -v or --view_before_saving. Please use --help for more information.')

    if (not args.style_image_path):
        parser.error("Error: Style image path not supplied. Please use --help for more information.")
    if (not args.content_image_path):
        parser.error("Error: Content image path not supplied. Please use --help for more information.")
    if (args.view_before_saving == None and args.epochs == None):
        parser.error("Error: Neither Epochs nor View Before Saving was specified. Please use --help for more information.")


    style_img = image_loader(args.style_image_path)
    content_img = image_loader(args.content_image_path)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    plt.ion()

    input_img = content_img.clone()
    if (args.view_before_saving):
        to_save = False
        output = 1
        while (not to_save):
            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img,num_steps=50)
            plt.figure()
            imshow(output, title='Output Image')
            while (1):
                response = input("Save the image as it is? yes or no\n")
                if (response.lower() == "yes"):
                    to_save = True
                    break
                elif (response.lower() == "no"):
                    to_save = False
                    break
                else:
                    print("Answer not recognized, asking again...")
        output_img = output.squeeze(0)      # remove the fake batch dimension
        output_img = unloader(output_img)
        output_img_path = "./out/output_"+str(os.path.basename(args.content_image_path).split(".")[0])+"_"+str(os.path.basename(args.style_image_path).split(".")[0])+".jpg"
        output_img.save(output_img_path)
    else:
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img,num_steps=int(args.epochs))
        
        output_img = output.squeeze(0)      # remove the fake batch dimension
        output_img = unloader(output_img)
        output_img_path = "./out/output_"+str(os.path.basename(args.content_image_path).split(".")[0])+"_"+str(os.path.basename(args.style_image_path).split(".")[0])+".jpg"
        output_img.save(output_img_path)

if __name__ == '__main__':
    main()