

import numpy as np
import matplotlib.pyplot as plt

def return_equidistal_xyz(num_points, r):
    """Algorithm to calculate num_points equidistial points on the surface of
    a sphere. Note that the returned number of points might slightly deviate
    from expected number of points.

    Algorith from: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    a = 4 * np.pi / num_points

    d = np.sqrt(a)
    M_theta = int(np.round(np.pi / d))

    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    xs = []
    ys = []
    zs = []

    i = 0
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            i += 1
            # if i == num_points:
            #     return xs, ys, zs
    return np.array(xs), np.array(ys), np.array(zs)


if __name__ == '__main__':

    num_points = 600
    r = 0.01
    xs, ys, zs = return_equidistal_xyz(num_points, r)
    print(num_points, len(zs))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='auto')
    ax.scatter(xs, ys, zs, c='r', marker='o')
    plt.show()