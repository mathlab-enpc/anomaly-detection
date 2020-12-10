import os
from density_computation import density_computation
import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt

def get_density_heatmaps_from_scene(scene_folder):
    densities_scene = []
    for t in range(len(glob.glob(os.path.join(scene_folder1, "*")))):
        image = Image.open(os.path.join(scene_folder1, "{}.png".format(t)))
        pixels = np.asarray(image)
        densities_scene.append(density_computation(pixels))
        print(densities_scene)
        return densities_scene

def sobolev_distance(bitmap1, bitmap2, gamma):
    n, p = bitmap1.shape
    s = 0.
    fft_bitmap_1 = np.fft.fft2(bitmap1)
    fft_bitmap_2 = np.fft.fft2(bitmap2)
    print(fft_bitmap_1)

    for i in range(n):
        for j in range(p):
            s += (1 + np.linalg.norm(np.array([i, j]))**2)**gamma * abs(fft_bitmap_1[i, j] - fft_bitmap_2[i, j])**2
    return np.sqrt(1./((n*p)**2) * s)


if __name__ == '__main__':


    scene_folder1 = os.path.join('experiment_images', 'parade_sequence')
    scene_folder2 = os.path.join('experiment_images', 'obstacle_sequence')

    print("computing densities")

    densities_scene_1 = get_density_heatmaps_from_scene(scene_folder1)
    densities_scene_2 = get_density_heatmaps_from_scene(scene_folder2)

    print("done!")

    distances = []
    number_of_frames_in_scene = min(len(densities_scene_1), len(densities_scene_2))

        
    for k in range(number_of_frames_in_scene):
        distance = sobolev_distance(densities_scene_1[k], densities_scene_2[k], 0.99)
        print(distance)
        distances.append(distance)

    plt.plot(range(number_of_frames_in_scene), distances)
    plt.show()











