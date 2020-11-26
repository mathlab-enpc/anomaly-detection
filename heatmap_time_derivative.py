import numpy as np
from PIL import Image
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt
import glob
import argparse
import os
from density_computation import density_computation

parser = argparse.ArgumentParser(description='Derivative w.r.t time')
parser.add_argument('--image-folder', type=str, default=os.path.join('experiment_images', 'natural_sequence'),
                    help="folder where images are located")
parser.add_argument('--output-folder', type=str, default=os.path.join('density_first_derivative', 'natural_sequence'),
                    help='name of the output file')
args = parser.parse_args()

if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)

def compute_derivative(matrix1, matrix2):
    output_matrix = matrix2 - matrix1
    return output_matrix
    
if __name__ == '__main__':
    for t in range(len(glob.glob(os.path.join(args.image_folder, '*')))-1):
        image1 = Image.open(os.path.join(args.image_folder, "{}.png".format(t)))
        pixels1 = np.asarray(image1)
        densities1 = density_computation(pixels1)
        image2 = Image.open(os.path.join(args.image_folder, "{}.png".format(t+1)))
        pixels2 = np.asarray(image2)
        densities2 = density_computation(pixels2)
        sns.heatmap(compute_derivative(densities1, densities2), vmax=2, vmin=-2)
        plt.savefig(os.path.join(args.output_folder, "{}.png".format(t)))
        plt.clf()
