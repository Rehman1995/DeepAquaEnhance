import os
import cv2
import numpy as np
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uqim_utils import getUIQM
#from brisque import BRISQUE
 
def uciqe(loc):
    img_bgr = cv2.imread(loc)        # Used to read image files
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]  # Coefficients obtained from the paper

    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))              # Chroma

    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(np.abs(1 - np.square(aver_chr / img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    quality_val = coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat         # get final quality value
    return quality_val


def measure_UIQMs(dir_name, im_res=(256,256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

def main():
    
    directory_path =  "/home/hasan/gan_mula/MuLA_GAN-main/EUVP_output8/"
    #directory_path ="/home/hasan/gan_mula/MuLA_GAN-main/UIEB_output7/"
    #directory_path ="/home/hasan/gan_mula/MuLA_GAN-main/UFO_output7/"
    #directory_path ="/home/hasan/gan_mula/MuLA_GAN-main/LSUI_output7/"
   # directory_path = "/Users/Rehman/Desktop/Khalifa/Research/Acquaculture review/Image Enhancement/Paper1/Outputs of Each/pugan/LSUI"
    uciqe_values = []
    uiqm_values = []
    niqe_values = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory_path, filename)
            uciqe_val = uciqe(img_path)
            uciqe_values.append(uciqe_val)
            #print(f"{filename}: UCIQE = {uciqe_val}")

            img = Image.open(img_path).resize((256, 256))
            uiqm_val = getUIQM(np.array(img))
            uiqm_values.append(uiqm_val)
            #print(f"{filename}: UIQM = {uiqm_val}")

            #img_gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            #niqe_val = BRISQUE(img_gray)
            #niqe_values.append(niqe_val)
            #print(f"{filename}: NUCIQEIQE = {niqe_val}")

    average_uciqe = np.mean(uciqe_values)
    average_uiqm = np.mean(uiqm_values)
    #average_niqe = np.mean(niqe_values)
    
    print(f"Average UCIQE: {average_uciqe}")
    print(f"Average UIQM: {average_uiqm}")
    #print(f"Average NIQE: {average_niqe}")



if __name__ == "__main__":
    main()
