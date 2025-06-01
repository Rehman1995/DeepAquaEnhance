
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

## options
parser = argparse.ArgumentParser()

#parser.add_argument("--data_dir", type=str, default="./Dataset/LSUI/test/images/")
#parser.add_argument("--sample_dir", type=str, default="./LSUI_output7")
#parser.add_argument("--model_name", type=str, default="MuLA_GAN_Initial_model_1")
#parser.add_argument("--weights_path", type=str, default="checkpoints/LSUI/generator_350.pth")


parser.add_argument("--data_dir", type=str, default="./Dataset/UIEB/test/images/")
#parser.add_argument("--sample_dir", type=str, default="./EUVP_output8")
parser.add_argument("--sample_dir", type=str, default="./UIEB_output7")
#parser.add_argument("--model_name", type=str, default="Newbkab4_MuLA_GAN_Initial_model_1")
#parser.add_argument("--weights_path", type=str, default="/home/gan_mula/MuLA_GAN-main/Newbkab4/UIEB/generator_320.pth")
#parser.add_argument("--weights_path", type=str, default="EUVP_proposed/EUVP/generator_140.pth")
parser.add_argument("--model_name", type=str, default="Newbkab3_MuLA_GAN_Initial_model_1")
parser.add_argument("--weights_path", type=str, default="/home/gan_mula/MuLA_GAN-main/Newbkab3/UIEB/generator_300.pth")


opt = parser.parse_args()

## checks
assert exists(opt.weights_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

# print(opt.model_name)
## model arch
if opt.model_name=='Newbkab3_MuLA_GAN_Initial_model_1': #'MuLA_GAN2':
    from models import Newbkab3_MuLA_GAN_Initial_model_1  #MuLA_GAN2
    model = Newbkab3_MuLA_GAN_Initial_model_1.MuLA_GAN_Generator()

## load weights
model.load_state_dict(torch.load(opt.weights_path))
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.weights_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)
## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    
    #print(f"Input image dimensions: {inp_img.shape}")  ##
    
    s = time.time()
    gen_img = (model(inp_img))
    times.append(time.time()-s)
    print("input image",inp_img.shape)
    print("generated image",gen_img.shape)
    
    gen_img_r = F.interpolate(gen_img, size=inp_img.shape[2:], mode='bilinear', align_corners=False)

    
    img_sample = torch.cat((inp_img.data, gen_img_r.data), -1)
    save_image(gen_img,join(opt.sample_dir, basename(path)), normalize=True)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    save_image(gen_img,join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))



