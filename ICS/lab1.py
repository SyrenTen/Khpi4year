import numpy as np
import matplotlib.pyplot as plt


def algorithm_1():
    dt = 0.01
    t = 0
    n = 0
    x0 = 0
    x = x0
    x1 = x0
    x2 = x0
    u = 0

    t_vals = []
    x_vals = []
    u_vals = []

    while t < 10:
        x_star = 0.01 * t ** 2

        if n < 2:
            delta_u = 0.1
        else:
            delta_u = ((x_star - 2 * x + x1) / (x - 2 * x1 + x2)) * delta_u

        u += delta_u
        x2 = x1
        x1 = x
        x += (0.1 * x ** 2 + u) * dt

        n += 1
        t = n * dt

        t_vals.append(t)
        x_vals.append(x)
        u_vals.append(u)

    plt.figure()
    plt.plot(t_vals, x_vals, label='x(t)')
    plt.plot(t_vals, u_vals, label='u(t)')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Algorithm 1: x(t) and u(t)')
    plt.legend()
    plt.grid()
    plt.show()


def algorithm_2():
    dt = 0.01
    t = 0
    n = 0
    x = -30

    t_vals = []
    x_vals = []
    x_star_vals = []
    u_star_vals = []

    while t < 100:
        x_star = 0.01 * t ** 2 - 30
        u_star = 0.02 * t - 0.1 * (0.01 * t ** 2 - 30) ** 2

        x += (0.1 * x ** 2 + u_star) * dt

        n += 1
        t = n * dt

        t_vals.append(t)
        x_vals.append(x)
        x_star_vals.append(x_star)
        u_star_vals.append(u_star)

    plt.figure()
    plt.plot(t_vals, x_star_vals, label='x*(t)')
    plt.plot(t_vals, u_star_vals, label='u*(t)')
    plt.plot(t_vals, x_vals, label='x(t)')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Algorithm 2: x*(t), u*(t), and x(t)')
    plt.legend()
    plt.grid()
    plt.show()


algorithm_1()
algorithm_2()
