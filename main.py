from functions import *

average_speedx = 10
speed_std = int(average_speedx)
trans_speed_std = int(average_speedx/3)
width, height = 200, 150
density = int(width/5)
time = 20
time_length_factor = time*2

""" Each of the following functions returns a bit array
of size width * height * time
And displays the corresponding map for each t=0,...,time
Uncomment as needed
"""

# no organized direction, "lobby" constraint
# 1st qrgument is always density
natural = natural_sequence(100, time, width, height)

# marching, same direction - "perfect" sample
translated = translated_sequence(density, time, width, height)

# marching, a few outliers - orthogonal or opposite directions
outliers_and_translation = translated_sequence_with_outliers(density, time, width, height)


# not necessarily relevant but can still be useful
# gathering people from instant 5 arount coordinates x=100, y=75
# in radius <= 8 and not too close after radius 5 (safety radius)
gathering = translated_sequence_gathering(density, 20, width, height, 5, 100, 75, 8)

# dispersion function to fix

# Density variations
# arguments: density, time, width, height, xh, yh, rh, speedh, increase_rate
# static hole
moving_hole_translated_sequence(300, 8, width, height, 100, 75, 30, 0,0)
# moving hole
moving_hole_translated_sequence(300, 8, width, height, 100, 75, 30, average_speedx,0)
# inflating hole
moving_hole_translated_sequence(300, 8, width, height, 60, 75, 20, average_speedx, 6)
# resorbing hole
moving_hole_translated_sequence(300, 8, width, height, 60, 75, 50, average_speedx, -20)

# speeding up subset - leads to local increase/decrease behind it
# bottom left
speed_varying_subset_translated_sequence(500, 5, width, height, 10, 10, 20, 20, 25)
# slowing down - small density hole beyond
speed_varying_subset_translated_sequence(600, 5, width, height, 35, 35, 25, average_speedx, -5)

# Faster single file - barely observable with high density
single_file_sequence(200, 10, width, height, 10, 30)

# Human row
row_sequence(300, 10, width, height, 10, 30)




