import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
out_img_num = 1000
for i in range(out_img_num):
	print("Colorization image {0}".format(i+1))
	opt.img_path = "./imgs/val_256/Places365_val_" + str(i+1).zfill(8) + ".jpg"
	img_gray_path = "./imgs/val_256_gray/Places365_val_" + str(i+1) + ".jpg"
	img = load_img(opt.img_path,img_gray_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	plt.imsave('./imgs_out/Places365_val_%s.jpg'%str(i+1), out_img_eccv16)

'''
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.show()
'''


