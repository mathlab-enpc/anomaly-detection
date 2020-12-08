import pyNetLogo

from PIL import Image

from point import Point
from functions import bit_matrix, bitmap


if __name__ == '__main__':
    netlogo = pyNetLogo.NetLogoLink(gui=False)
    netlogo.load_model("netlogo_models/Flocking.nlogo")
    netlogo.command("setup")


    step = 100

    max_x, min_x = netlogo.report("min-pxcor"), netlogo.report("max-pxcor")
    max_y, min_y = netlogo.report("min-pycor"), netlogo.report("max-pycor")
    Lx, Ly = 300, 300

    mat = []

    for k in range(step):

        netlogo.command("go")
        x = netlogo.report("map [s -> [xcor] of s] sort turtles")
        y = netlogo.report("map [s -> [ycor] of s] sort turtles")
        x = ((x - min_x) / (max_x - min_x) * Lx).astype(int)
        y = ((y - min_y) / (max_y - min_y) * Ly).astype(int)
        points = [[] for k in range(len(x))]
        for k, (x_, y_) in enumerate(zip(x, y)):
            points[k] = Point(k, x_, y_, 0, 0)
        mat.append(bit_matrix(points, Lx, Ly))

    folder_name = "netlogo_simul"
    for t, m in enumerate(mat):
        bitmap(m, t, folder_name)
    
    print('done !')
    netlogo.kill_workspace()

