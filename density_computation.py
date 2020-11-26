import numpy as np
from PIL import Image
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Density computation script')
parser.add_argument('--image-folder', type=str, default=os.path.join('experiment_images', 'natural_sequence'),
                    help="folder where images are located")
parser.add_argument('--output-folder', type=str, default=os.path.join('density_experiment', 'natural_sequence'),
                    help='name of the output file')
args = parser.parse_args()

if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)

def density_computation(matrix):
    n, p = matrix.shape
    coordinates = []
    for i in range(n):
        for j in range(p):
            if matrix[i, j] == 1:
                coordinates.append(np.array([i, j]))

    local_densities = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for x in coordinates:
                local_densities[i, j] += np.exp(-(np.linalg.norm(np.array([i, j]) - x)/10)**2)

    return local_densities
    
if __name__ == '__main__':
    for t in range(len(glob.glob(os.path.join(args.image_folder, "*")))):
        image = Image.open(os.path.join(args.image_folder, "{}.png".format(t)))
        pixels = np.asarray(image)
        densities = density_computation(pixels)
        sns.heatmap(densities, vmax=14)
        plt.savefig(os.path.join(args.output_folder, "{}.png".format(t)))
        plt.clf()
