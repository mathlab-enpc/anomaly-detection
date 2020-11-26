import numpy as np
from PIL import Image
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt
import glob
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='Speed computation script')
parser.add_argument('--image-folder', type=str, default=os.path.join('experiment_images', 'natural_sequence'),
                    help="folder where images are located")
parser.add_argument('--output-folder', type=str, default=os.path.join('speed_experiment', 'natural'),
                    help='name of the output file')
args = parser.parse_args()

if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)

def speed_computation(matrix1, matrix2):
    n, p = matrix1.shape
    coordinates_matrix1 = []
    coordinates_matrix2 = []

    for i in range(n):
        for j in range(p):
            if matrix1[i, j] == 1:
                coordinates_matrix1.append(np.array([i, j]))
            if matrix2[i, j] == 1:
                coordinates_matrix2.append(np.array([i, j]))

    number_coord1, number_coord2 = len(coordinates_matrix1), len(coordinates_matrix2)
    distances = np.zeros((number_coord1, number_coord2))
    for i in range(number_coord1):
        for j in range(number_coord2):
            distances[i, j] = np.linalg.norm(coordinates_matrix1[i] - coordinates_matrix2[j])

    corresponding_points = np.argmin(distances, axis=0)
    speeds = np.zeros((n, p, 2))
    for i in range(number_coord2):
        speeds[coordinates_matrix2[i][0], coordinates_matrix2[i][1]] = coordinates_matrix2[i] - coordinates_matrix1[corresponding_points[i]]

    return speeds
    
if __name__ == '__main__':
    for t in range(len(glob.glob(os.path.join(args.image_folder, '*')))-1):
        image1 = Image.open(os.path.join(args.image_folder, "{}.png".format(t)))
        image2 = Image.open(os.path.join(args.image_folder, "{}.png".format(t+1)))
        pixels1 = np.asarray(image1)
        pixels2 = np.asarray(image2)
        speed = speed_computation(pixels1, pixels2)
        n, p, _ = speed.shape
        output_image = cv2.imread(os.path.join(args.image_folder, "{}.png".format(t+1)))
        color = (0, 255, 0)
        thickness = 1
        
        for i in range(n):
            for j in range(p):
                if np.linalg.norm(speed[i, j]) > 0:
                    start_point = (i, j)
                    end_point = (int(round(i + speed[i, j][0], 0)), int(round(j + speed[i, j][1], 0)))
                    image = cv2.arrowedLine(output_image, start_point, end_point, 
                        color, thickness)

        cv2.imwrite(os.path.join(args.output_folder, "{}.png".format(t)), output_image) 