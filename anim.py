# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:27:12 2020

@author: paulm
"""

from matplotlib import animation, rc
from matplotlib import pyplot as plt

"""Fonctions d'animations"""

def init():
    """Fonction qui plot l'arrière-plan/trucs qui change pas"""
    #exemple : plt.plot(les murs rouges)
    #attention il faut que ça plot au bon endroit donc mets un ax=ax dans le plot
    n = width
    p = height
    abscissas = np.array([i for i in range (n)])
    y0_lobby = np.array([0 for i in range (n)])
    ymax_lobby = np.array([p - 1 for i in range (n)])
    plt.plot(abscissas, y0_lobby, color = "r")
    plt.plot(abscissas, ymax_lobby, color = "r")

def natural_sequence_anim(density, time, width, height):
    points = initial_points(density, width, height)
    new_points = points
    inception = np.zeros((width,height,time))
    matrix0 = bit_matrix(points, width, height)
    inception[:,:,0] = matrix0
    bitmap(matrix0)
    for t in range (time):
        new_points = naturally_evolving_points(new_points, width, height)
        new_matrix = bit_matrix(new_points, width, height)
        inception[:,:,t] = new_matrix
        bitmap(new_matrix)
    return inception

test = natural_sequence_anim(density, time, width, height)

def coordinates(test):
    coor = np.zeros((len(test)*2,2, time))
    for t in range (time):
        index = 0
        for i in range (width):
            for j in range (height):
                if test[i,j,t] == 1:
                    coor[index,0,t] = i
                    coor[index,1,t] = j
                    index += 1
    return coor

test2 = coordinates(test)

def animate(i):
    """Fonction qui plot l'image numéro i"""
    """Si tu connais les coordonnées des points au temps i tu fais :"""
    #plt.plot(points[i], ax=ax)
    plt.plot( test2[i,:,:][0], test2[i,:,:][1], "x")

"""Initialisation de la figure"""

fig=plt.figure()
ax=plt.axes()
ax.set_aspect('equal')
ax.set_axis_off()

"""Animation en elle-même"""


###################### TO FIX - pb = probs test2
anim=animation.FuncAnimation(fig, animate, init_func=init, frames=20, interval=1000)
rc('animation',html='html5')
anim

"frames: nombres d'images que tu veux, i va de 0 à n_frames-1"
"interval: durée entre deux images, je crois que c'est en ms"