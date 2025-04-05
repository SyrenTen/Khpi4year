import numpy as np
import matplotlib.pyplot as plt


def lab2():
    t = 0
    n = 0
    dt = 0.001
    s = 0
    bl = 0.05
    b0 = 0.0006
    x = 1
    x1 = 0
    k = 100

    x1_star = 0
    x_star = 0

    t_vals = []
    x_star_vals = []
    u_vals = []
    x_vals = []

    while t < 60:
        a1 = 0.5 + 0.2 * np.sin(((2 * np.pi) / 100) * t)
        a0 = 0.06 + 0.1 * np.cos(((2 * np.pi) / 130) * t)

        u = k * s
        x2 = u - a1 * x1 - a0 * x
        x2_star = -bl * x1 - b0 * x

        x = x + x1 * dt
        x1 = x1 + x2 * dt
        s = s + (x2_star - x2) * dt

        x2_star = -bl * x1_star - b0 * x_star
        x_star = x_star + x1_star * dt
        x1_star = x1_star + x2_star * dt

        n += 1
        t = n * dt

        x_star_vals.append(x_star)
        u_vals.append(u)
        x_vals.append(x)
        t_vals.append(t)

    plt.figure()
    plt.plot(t_vals, x_star_vals, label='x*(t)')
    plt.plot(t_vals, u_vals, label='u*(t)')
    plt.plot(t_vals, x_vals, label='x(t)')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Lab2')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    lab2()
