import cv2
import numpy as np
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Video creation script')
parser.add_argument('--image-folder', type=str, default=os.path.join('experiment_images', 'natural_sequence'),
                    help="folder where images are located")
parser.add_argument('--video-filename', type=str, default='experiment',
                    help='name of the output file')
args = parser.parse_args()
img_array = []

if not os.path.isdir('videos'):
    os.makedirs('videos')

for t in range(len(glob.glob(os.path.join(args.image_folder, "*")))):
    img = cv2.imread(os.path.join(args.image_folder, "{}.png".format(t)))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(
    os.path.join('videos', "{}.mp4".format(args.video_filename)),
    cv2.VideoWriter_fourcc(*'mp4v'),
    3,
    size
)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
