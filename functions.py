from point import Point
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

average_speedx = 10
speed_std = int(average_speedx)
trans_speed_std = int(average_speedx/3)
width, height = 200, 150
density = int(width/5)
time = 20
time_length_factor = time*2

"""
Different kinds of outliers
1. Basic functions - random motion in a lobby-like frame
2. Translation in the same direction, crossing people or not
3. Scattering - useless
4. Gathering
5. Density variations via intentional holes
6. Speeding up/slowing down subsets
7. Single file/human row
8. TBD - Group creation?
9. TBD - density variation with time?
"""

""" Section 1: basic functions
Random motion in a lobby-like frame
"""
# list of points given total density
def initial_points(density, width, height):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    for i in range (density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = max( average_speedx/4, round(np.random.normal(average_speedx, speed_std)) )
        if np.random.randint(2) == 0:
            speedx = - speedx
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

# returns bit matrix of points
def bit_matrix(points, width, height):
    matrix = np.zeros((width, height))
    for i in range (len(points)):
        x = points[i].x0
        y = points[i].y0
        if x < width and x >= 0 and y < height and y >= 0:
            matrix[x,y] = 1
    return matrix

# displays bit matrix
def bitmap(matrix, t, folder_name):
    n, p = matrix.shape
    img = Image.new("1", (n, p))
    pixels = img.load()
    for i in range(n):
        for j in range(p):
            if matrix[i, j] == 1:
                pixels[i, j] = 1
    if not os.path.isdir('experiment_images'):
        os.makedirs('experiment_images')
    if not os.path.isdir(os.path.join('experiment_images', folder_name)):
        os.makedirs(os.path.join('experiment_images', folder_name))
    img.save(os.path.join('experiment_images', folder_name, "{}.png".format(t)))

# some points can be out, handled in bit matrix
def naturally_evolving_points(points, width, height):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if y > height:
            y = height
            #speedy = - speedy
            speedy = 0
        if y < 0:
            y = 0
            speedy = - speedy
            speedy = 0
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points
""" Poisson distr - to figure out?
    # poisson distribution to add points
    # parameters to adjust
    add = int(np.random.poisson(2))
    for i in range (add):
        x = np.random.randint(width)
        y = np.random.randint(height)
        x_at_zero = np.random.randint(4)
        if x_at_zero == 0:
            x = 0
        if x_at_zero == 1:
            x = width-1
        if x_at_zero == 2:
            y = 0
        else:
            y = height-1
        speedx = round(np.random.normal(average_speedx, speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(len(points), x, y, speedx, speedy)
        points = np.append(points, p)
"""


def natural_sequence(density, time, width, height):
    points = initial_points(density, width, height)
    new_points = points
    #inception = np.array([])
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    #inception = np.append(inception, matrix0)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "natural_sequence")
    for t in range (time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        #inception = np.append(inception, new_matrix)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "natural_sequence")
    return inception

# l = natural_sequence(density,3,width, height)
# is a list of bit matrix, index i being its state at time i
#natural_sequence(density,20,width, height)


""" Section 2:
Translated motion in a lobby-like frame
"""

def initial_trans_points(density, width, height):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    for i in range (density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def translated_sequence(density, time, width, height):
    points = initial_trans_points(density, width, height)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "translated_sequence")
    for t in range (time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence")
    return inception

#translated_sequence(density,20,width, height)

outlier_xproportion = 13
# 1 outlier for 15 points
outlier_yproportion = 8

def initial_trans_points_with_outliers(density, width, height):
    points = np.array([])
    
    # black block
    x = np.random.randint(-width,width)
    y = np.random.randint(height)
    speedx = round(np.random.normal(0, 2))
    speedy = np.random.randint(-average_speedx/2,average_speedx/2) 
    p = Point(0, x, y, speedx, speedy)
    points = np.append(points, p)
    
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    for i in range (1,density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        if i % outlier_xproportion == 0:
            speedx = -speedx
        if i % outlier_yproportion == 0:
            speedy = np.random.randint(-average_speedx/2,average_speedx/2)
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def translated_sequence_with_outliers(density, time, width, height):
    points = initial_trans_points_with_outliers(density, width, height)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "translated_sequence_with_outliers")
    for t in range (time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence_with_outliers")
    return inception

#translated_sequence_with_outliers(density, 20, width, height)

""" Section 3:
Scattering
xg, yg are the coordinates of the disturbing element, rg its "action radius"
"""

def scattering_points(points, width, height, xg, yg, rg):
    rs = average_speedx/3
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if y > height:
            y = height
            speedy = - speedy
            speedy = 0
        if y < 0:
            y = 0
            speedy = - speedy
            speedy = 0
        d = np.sqrt( (x-xg)**2 + (y-yg)**2 )
        if d < rg**2:
            if d < rs**2:
                speedx = 0
                speedy = 0
            else:
                speed0 = np.sqrt(speedx**2 + speedy**2)
                speedx = int((x-xg)*speed0/d)
                speedy = int((y-yg)*speed0/d)
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def translated_sequence_scattering(density, time, width, height, trigger_time, xg, yg, rg):
    points = initial_trans_points(density, width, height)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "translated_sequence_scattering")
    for t in range (trigger_time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence_scattering")
    new_points = np.append(new_points, Point(-1,xg,yg,0,0))
    for t in range (trigger_time, time):
        new_points = scattering_points(new_points, width, height, xg, yg, rg)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence_scattering")
    return inception

# doesn't work properly
#translated_sequence_scattering(density, 20, width, height, 5, 100, 75, 7)

""" Section 4:
Gathering
xg, yg are the coordinates of the disturbing element, rg its "action radius"
"""
def gathering_points(points, width, height, xg, yg, rg):
    rs = average_speedx/3
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if y > height:
            y = height
            speedy = - speedy
            speedy = 0
        if y < 0:
            y = 0
            speedy = - speedy
            speedy = 0
        d = np.sqrt( (x-xg)**2 + (y-yg)**2 )
        if d < rg**2:
            if d < rs**2:
                speedx = 0
                speedy = 0
            else:
                speed0 = np.sqrt(speedx**2 + speedy**2)
                speedx = int((xg-x)*speed0/d)
                speedy = int((yg-y)*speed0/d)
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def translated_sequence_gathering(density, time, width, height, trigger_time, xg, yg, rg):
    points = initial_trans_points(density, width, height)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "translated_sequence_gathering")
    for t in range (trigger_time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence_gathering")
    new_points = np.append(new_points, Point(-1,xg,yg,0,0))
    for t in range (trigger_time, time):
        new_points = gathering_points(new_points, width, height, xg, yg, rg)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "translated_sequence_gathering")
    return inception

#translated_sequence_gathering(density, 20, width, height, 5, 100, 75, 7)
    

""" Section 5:
Intentional holes/density variations
xh, yh are the hole coordinates, rh its radius, speedh its horizontal speed
increase_rate is the expansion factor of the hole - can be negative
"""

def keep_out(point, xh, yh, rh):
    x = point.x0
    y = point.y0
    distance = np.sqrt((x-xh)**2 + (y-yh)**2)
    if distance <= rh:
        adjust = np.sqrt(rh**2 - (x-xh)**2)
        if y >= yh:
            y = yh + adjust
        else:
            y = yh - adjust
        y = int(y)
    p = Point(point.index, x, y, point.speedx, point.speedy)
    return p

def keep_out_sequence(points, xh, yh, rh):
    for i in range (len(points)):
        points[i] = keep_out(points[i], xh, yh, rh)
    return points

def moving_hole_translated_sequence(density, time, width, height, xh, yh, rh, speedh, increase_rate):
    points = initial_trans_points(density, width, height)
    new_points = keep_out_sequence(points, xh, yh, rh)
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "moving_hole_translated_sequence")
    for t in range (time):
        increase = int(t*increase_rate)
        new_points = naturally_evolving_points(new_points, width, height)
        new_points = keep_out_sequence(new_points, xh+t*speedh, yh, max(0,rh+increase))
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "moving_hole_translated_sequence")
    return inception

#moving_hole_translated_sequence(300, 8, width, height, 100, 75, 30, 0, 0)
#moving_hole_translated_sequence(300, 8, width, height, 100, 75, 30, average_speedx, 0)
#moving_hole_translated_sequence(300, 8, width, height, 60, 75, 20, average_speedx, 6)
#moving_hole_translated_sequence(300, 8, width, height, 60, 75, 50, average_speedx, -20)

""" Section 6:
Speeding up/slowing down subsets
implying local density variations
- for now one subset, (xv,yv) radius rv and speed speedv
"""

def initial_trans_density_varying_points(density, width, height, xv, yv, rv, speedv):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    for i in range (density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        d = np.sqrt((x-xv)**2 + (y-yv)**2)
        if d <= rv:
            speedx = speedv
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def varying_naturally_evolving_points(points, width, height, xv, yv, rv, speedv, acceleration):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if y >= height:
            y = height-1
            speedy = 0
        if y < 0:
            y = 0
            speedy = 0
        d = np.sqrt((p.x0-xv)**2 + (p.y0-yv)**2)
        if d <= rv and p.speedx == speedv-acceleration:
            x = p.x0 + int(max(1, speedv))
            speedx = speedv
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def speed_varying_subset_translated_sequence(density, time, width, height, xv, yv, rv, speedv, acceleration):
    points = initial_trans_density_varying_points(density, width, height,
                                                  xv, yv, rv, speedv)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "speed_varying_subset_translated_sequence")
    for t in range (time):
        new_points = varying_naturally_evolving_points(new_points, width, height,
                                                       xv, yv, rv, speedv + acceleration, acceleration)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "speed_varying_subset_translated_sequence")
    return inception

#speed_varying_subset_translated_sequence(500, 5, width, height, 10, 10, 20, 20, 25)
#speed_varying_subset_translated_sequence(600, 5, width, height, 35, 35, 25, average_speedx, -5)


""" Section 7:
Single file or human row
"""

def single_file_initial_trans_points(density, width, height, xf, yf):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    #single_file_size = np.random.randint(5,11)
    single_file_size = 10
    speedf = max( average_speedx, round(np.random.normal(average_speedx*2, speed_std)) )
    for i in range (single_file_size):
        x = xf - 3 * (i+1)
        y = yf
        speedx = speedf
        speedy = 0
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    for i in range (single_file_size, density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def single_file_naturally_evolving_points(points, width, height):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if i > 0 and p.speedx == points[i-1].speedx:
            y = points[i-1].y0
        if y > height:
            y = height
            speedy = 0
        if y < 0:
            y = 0
            speedy = 0
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def single_file_sequence(density, time, width, height, xf, yf):
    points = single_file_initial_trans_points(density, width, height, xf, yf)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "single_file_sequence")
    for t in range (time):
        new_points = single_file_naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "single_file_sequence")
    return inception

#single_file_sequence(200, 10, width, height, 10, 30)


def row_initial_trans_points(density, width, height, xr, yr, speedr):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    #row_size = np.random.randint(5,11)
    row_size = 10
    for i in range (row_size):
        if yr + 3 * row_size > height:
            y = yr - 3 * (i+1)
        else:
            y = yr + 3 * (i+1)
        x = xr
        speedx = speedr
        speedy = 0
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    for i in range (row_size, density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def row_naturally_evolving_points(points, width, height):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if i > 0 and p.speedx == points[i-1].speedx:
            x = points[i-1].x0
        if y > height:
            y = height
            speedy = 0
        if y < 0:
            y = 0
            speedy = 0
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points


def row_sequence(density, time, width, height, xr, yr):
    speedr = round(np.random.normal(average_speedx, trans_speed_std))
    points = row_initial_trans_points(density, width, height, xr, yr, speedr)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "row_sequence")
    for t in range (time):
        new_points = row_naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, "row_sequence")
    return inception

#row_sequence(300, 10, width, height, 10, 30)

""" Section 8:
Marathon setup
Runners in the middle, compact crowd around
"""

def initial_marathon_points(density, width, height, runners_proportion, epsilon):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    epsilon = 1/8
    n_runners = int(runners_proportion*density)
    # runners initialization
    for i in range (n_runners):
        x = np.random.randint(-w,w)
        y = np.random.randint(int(height*epsilon), int(height*(1-epsilon)))
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    # surrounding crowd initialization
    for i in range (n_runners, density):
        x = np.random.randint(-w,w)
        y = np.random.randint(int(height*epsilon))
        top = np.random.randint(2)
        if top == 1:
            y = np.random.randint(int(height*(1-epsilon)), height)
        speedx = 0
        speedy = 0
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def is_in(index, table):
    for i in range (len(table)):
        if table[i] == index:
            return True
    return False

def marathon_evolving_points(points, width, height, followers, epsilon):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if is_in(i, followers[0,:]):
            if y >= height*(1-epsilon):
                y = y - np.random.randint(average_speedx)
                speedx = round(np.random.normal(average_speedx, 2))
            if y <= height*epsilon:
                y = y + np.random.randint(average_speedx)
                speedx = round(np.random.normal(average_speedx, 2))
            if followers[1,i] <= 0:
                speedx = max(0, speedx - average_speedx/2)
                if y < height*epsilon + average_speedx:
                    y = height*epsilon
                if y > height*(1-epsilon) - average_speedx:
                    y = height*(1-epsilon)
        if y > height:
            y = height
            speedy = 0
        if y < 0:
            y = 0
            speedy = 0
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points


def translated_marathon_sequence(density, time, width, height, runners_proportion, n_followers, epsilon):
    points = initial_marathon_points(density, width, height, runners_proportion, epsilon)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, "translated_marathon_sequence")
    n = int(n_followers)
    followers = np.zeros((2, n))
    for i in range (n):
        followers[0,i] = np.random.randint(n)
        followers[1,i] = round(np.random.normal(width, 20))
    for t in range (time):
        new_points = marathon_evolving_points(new_points, width, height, followers, epsilon)
        for i in range (n):
            followers[1,i] -= 1
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, 'translated_marathon_sequence')
    return inception

#translated_marathon_sequence(200, 10, width, height, 0.25, density/2, 1/8)

""" Section 9:
Obstacle
To circumvent
xo, yo, wo, ho are the (rectangle) obstacle coordinates and dimensions
"""

def initial_obstacle_points(density, width, height, xo, yo, wo, ho):
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    for i in range (density):
        x = np.random.randint(-w,w)
        y = np.random.randint(height)
        if abs(x-xo) < int(wo/2) and abs(y-yo) < int(ho/2):
            if y < yo:
                y = yo - int(ho/2)
            else:
                y = yo + int(ho/2)
        # "rounding corners"
        if np.sqrt( ( x - xo + int(wo/2))**2 + (y-yo)**2 ) < ho/2:
            if y < yo:
                y = yo - x + xo - int(wo/2 + ho/2)
            else:
                y = yo + x - xo + int(wo/2 + ho/2)
        speedx = round(np.random.normal(average_speedx, trans_speed_std))
        speedy = round(np.random.normal(0, 2))
        p = Point(i, x, y, speedx, speedy)
        points = np.append(points, p)
    return points

def obstacle_evolving_points(points, width, height, xo, yo, wo, ho):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + round(np.random.normal(p.speedx, 1))
        y = p.y0 + round(np.random.normal(p.speedy, 1))
        speedx = p.speedx
        speedy = p.speedy
        if y > height:
            y = height
            speedy = 0
        if y < 0:
            y = 0
            speedy = 0
        if abs(x-xo) < int(wo/2) and abs(y-yo) < int(ho/2):
            if y < yo:
                y = yo - int(ho/2)
            else:
                y = yo + int(ho/2)
        # "rounding corners"
        if np.sqrt( ( x - xo + int(wo/2))**2 + (y-yo)**2 ) < ho/2:
            if y < yo:
                y = yo - x + xo - int(wo/2 + ho/2)
            else:
                y = yo + x - xo + int(wo/2 + ho/2)
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def obstacle_sequence(density, time, width, height, xo, yo, wo, ho):
    points = initial_obstacle_points(density, width, height, xo, yo, wo, ho)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, 'obstacle_sequence')
    for t in range (time):
        new_points = obstacle_evolving_points(new_points, width, height, xo, yo, wo, ho)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, 'obstacle_sequence')
    return inception

#obstacle_sequence(density*7, 10, width, height, 70, 30, 60, 30)


""" Section 10:
Military parade
Very regular ranks and "breaks"
"""

def initial_parade_points(density, width, height, n_row, n_line, space):
    # spectators (6*2) + ongoing parade (8*6 + 1)
    # 5 pixels between each point, i.e. parade width = 30, length = 45, break= 20?
    points = np.array([])
    width *= time_length_factor
    density *= time_length_factor
    w = int(width/2)
    interval = (n_row + 1) * space * 2 # 2 = 1 + 1 for the break
    sitting_interval = 4 * space *3/2
    for i in range (-w,w):
        if i % interval < n_row * space and i % space == 0:
            x = i
            speedx = average_speedx
            speedy = 0
            for j in range (n_line):
                y = int(height/2) + space * (j - int(n_line/2))
                p = Point(i, x, y, speedx, speedy)
                points = np.append(points, p)
        if i % interval == (n_row + 2) * space:
            x = i
            speedx = average_speedx
            speedy = 0
            y = int((height - space)/2)
            p = Point(i, x, y, speedx, speedy)
            points = np.append(points, p)
        if i % sitting_interval < 4 * space and i % space == 0:
            x = i
            speedx = 0
            speedy = 0
            for j in range (2):
                y = (j+1) * space
                p = Point(i, x, y, speedx, speedy)
                points = np.append(points, p)
            for j in range (2):
                y = height - (j+1) * space
                p = Point(i, x, y, speedx, speedy)
                points = np.append(points, p)
    return points


def parade_evolving_points(points, width, height):
    for i in range (len(points)):
        p = points[i]
        x = p.x0 + p.speedx
        y = p.y0 + p.speedy
        speedx = p.speedx
        speedy = p.speedy
        new = Point(i, x, y, speedx, speedy)
        points[i] = new
    return points

def parade_sequence(density, time, width, height, n_row, n_line, space):
    points = initial_parade_points(density, width, height, n_row, n_line, space)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0, 0, 'parade_sequence')
    for t in range (time):
        new_points = parade_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix, t+1, 'parade_sequence')
    return inception

#parade_sequence(density, 10, width, height, 8, 6, 8)
