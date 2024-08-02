
"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""

%cd /home/hasan/gan_mula/MuLA_GAN-main/Evaluation/

## python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from imqual_utils import getSSIM, getPSNR


## compares avg ssim and psnr 
def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256,256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f==gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L"); g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)


"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
# gtr_dir = "/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/UEIB_Dataset/Data2/test/testB"

#gtr_dir = "/Users/Rehman/Desktop/Khalifa/Research/Acquaculture review/Image Enhancement/MuLA_GAN-main/MuLA_GAN-main/Dataset/UIEB/test/groundtruth"


#gtr_dir = "/home/hasan/gan_mula/MuLA_GAN-main/Dataset/EUVP/test/groundtruth/" 
gtr_dir = "/home/hasan/gan_mula/MuLA_GAN-main/Dataset/UIEB/test/groundtruth/" 
#gtr_dir = "/home/hasan/gan_mula/MuLA_GAN-main/Dataset/UFO_120/test/groundtruth/" 
#gtr_dir = "/home/hasan/gan_mula/MuLA_GAN-main/Dataset/LSUI/test/groundtruth/" 


#gtr_dir = "/home/hasan/gan_mula/MuLA_GAN-main/pugan/PUGAN_TIP2023/dataUIEB/tests/gt/"    
## generated im paths

#gen_dir = "/Users/Rehman/Desktop/Khalifa/Research/Acquaculture review/Image Enhancement/MuLA_GAN-main/MuLA_GAN-main/output" 

#gen_dir = "/home/hasan/gan_mula/MuLA_GAN-main/EUVP_output8/"
gen_dir = "/home/hasan/gan_mula/MuLA_GAN-main/UIEB_output7/"
#gen_dir = "/home/hasan/gan_mula/MuLA_GAN-main/UFO_output7/"
#gen_dir = "/home/hasan/gan_mula/MuLA_GAN-main/LSUI_output7/"


#gen_dir =  "/home/hasan/gan_mula/MuLA_GAN-main/pugan/PUGAN_TIP2023/samples_UIEB/"

#gen_dir = "eval_data/ufo_test/deep-sesr/" 


### compute SSIM and PSNR
SSIM_measures, PSNR_measures = SSIMs_PSNRs(gtr_dir, gen_dir)
print ("SSIM on {0} samples".format(len(SSIM_measures)))
print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

print ("PSNR on {0} samples".format(len(PSNR_measures)))
print ("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))



