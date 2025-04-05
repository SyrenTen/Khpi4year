import numpy as np
import matplotlib.pyplot as plt


def algorithm1():
    t_values = []
    x_values = []
    x1_values = []
    x_star_values = []
    x1_star_values = []
    u_star_values = []
    t = 0
    dt = 0.01
    T = 1
    x0 = 1
    x10 = 0
    xT = 0
    x1T = 0
    x = x0
    x1 = x10

    d = x0
    c = x10
    b = (3 * (xT - x0 - x10 * T) - T * (x1T - x10)) / T ** 2
    a = (T * (x1T - x10) - 2 * (xT - x0 - x10 * T)) / T ** 3

    while t < T:
        x_star = a * (t ** 3) + b * (t ** 2) + (c * t) + d
        x1_star = 3 * a * (t ** 2) + 2 * b * t + c
        u_star = 6 * a * t + 2 * b - (2.5 * (np.pi ** 2)) * (a * (t ** 3) + b * (t ** 2) + (c * t) + d)

        x2 = u_star + (2.5 * (np.pi ** 2)) * x
        x = x + x1 * dt
        x1 = x1 + x2 * dt

        t = t + dt

        t_values.append(t)
        x_values.append(x)
        x1_values.append(x1)
        x_star_values.append(x_star)
        x1_star_values.append(x1_star)
        u_star_values.append(u_star)

    plt.figure()
    plt.plot(t_values, x_values, label='x')
    plt.plot(t_values, x1_values, label='x1')
    plt.plot(t_values, x_star_values, label='x_star')
    plt.plot(t_values, x1_star_values, label='x1_star')
    plt.plot(t_values, u_star_values, label='u_star')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Algorithm 1')
    plt.legend()
    plt.grid()
    plt.show()


def algorithm2():
    t_values = []
    x_values = []
    x1_values = []
    x_star_values = []
    x1_star_values = []
    u_values = []

    t = 0
    dt = 0.01
    T = 1
    x0 = 1
    x10 = 0
    xT = 0
    x1T = 0
    x = x0
    x1 = x10

    d = x0
    c = x10
    b = (3 * (xT - x0 - x10 * T) - T * (x1T - x10)) / T ** 2
    a = (T * (x1T - x10) - 2 * (xT - x0 - x10 * T)) / T ** 3

    while t < T - dt:
        d1 = x
        c1 = x1
        b1 = (3 * (xT - x - x1 * (T - t)) - (T - t) * (x1T - x1)) / (T - t) ** 2
        a1 = ((T - t) * (x1T - x1) - 2 * (xT - x - x1 * (T - t))) / (T - t) ** 3

        x_star = a * (t ** 3) + b * (t ** 2) + (c * t) + d

        x1_star = 3 * a * (t ** 2) + 2 * b * t + c

        u = 2 * b1 - (2.5 * (np.pi ** 2)) * d1

        x2 = u + (2.5 * (np.pi ** 2)) * x
        x = x + x1 * dt
        x1 = x1 + x2 * dt

        t = t + dt

        t_values.append(t)
        x_values.append(x)
        x1_values.append(x1)
        x_star_values.append(x_star)
        x1_star_values.append(x1_star)
        u_values.append(u)

    plt.figure()
    plt.plot(t_values, x_values, label='x')
    plt.plot(t_values, x1_values, label='x1')
    plt.plot(t_values, x_star_values, label='x_star')
    plt.plot(t_values, x1_star_values, label='x1_star')
    plt.plot(t_values, u_values, label='u')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Algorithm 2')
    plt.legend()
    plt.grid()
    plt.show()


def algorithm3():
    t_values = []
    x_values = []
    x1_values = []
    x_star_values = []
    x1_star_values = []
    u_values = []

    t = 0
    dt = 0.01
    T = 1
    x0 = 1
    x10 = 0
    xT = 0
    x1T = 0
    x = x0
    x1 = x10
    delta_t = 0.1

    d = x0
    c = x10
    b = (3 * (xT - x0 - x10 * T) - T * (x1T - x10)) / T ** 2
    a = (T * (x1T - x10) - 2 * (xT - x0 - x10 * T)) / T ** 3

    while t < T:
        x_star = a * (t + delta_t) ** 3 + b * (t + delta_t) ** 2 + (c * (t + delta_t)) + d
        x1_star = 3 * a * (t + delta_t) ** 2 + 2 * b * (t + delta_t) + c

        d2 = x
        c2 = x1

        # b2 = (3 * (x * (t + delta_t) - x - x1 * delta_t) - delta_t * (x1_star * (t + delta_t) - x1)) / delta_t ** 2
        # a2 = (delta_t * (x1 * (t + delta_t) - x1) - 2 * (x_star * (t + delta_t) - x - x1 * delta_t)) / delta_t ** 3

        b2 = (3 * (x_star - x - x1 * delta_t) - delta_t * (x1_star - x1)) / delta_t ** 2
        a1 = (delta_t * (x1_star - x1) - 2 * (x_star - x - x1 * delta_t)) / delta_t ** 3

        u = 2 * b2 - (2.5 * (np.pi ** 2)) * d2

        x2 = u + (2.5 * (np.pi ** 2)) * x
        x = x + x1 * dt
        x1 = x1 + x2 * dt

        t = t + dt

        t_values.append(t)
        x_values.append(x)
        x1_values.append(x1)
        x_star_values.append(x_star)
        x1_star_values.append(x1_star)
        u_values.append(u)

    plt.figure()
    plt.plot(t_values, x_values, label='x')
    plt.plot(t_values, x1_values, label='x1')
    plt.plot(t_values, x_star_values, label='x_star')
    plt.plot(t_values, x1_star_values, label='x1_star')
    plt.plot(t_values, u_values, label='u')
    plt.xlabel('Time t')
    plt.ylabel('Values')
    plt.title('Algorithm 3')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    algorithm1()
    algorithm2()
    algorithm3()
