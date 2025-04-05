import numpy as np
import matplotlib.pyplot as plt

# Initial values
t = 0
dt = 0.01

x1 = 0.1
x1_1 = 1  # x1'
x2 = 1
x2_1 = 0.01  # x2'

w = 0.628
u1 = 0
u2 = 0

# Parameters (not all needed for variant 2)
v0 = 0.1  # from program trajectory x1_dot = v = 0.1

# Lists to save data
t_list = []
delta1_list = []
delta2_list = []
u1_list = []
u2_list = []

while t < 10:
    # Target accelerations for variant 2:
    x1_2_star = -x1 * x2 + u1
    x2_2_star = u2

    # Control laws
    u1 = x1_2_star - x1_1 - x2_1
    u2 = x2_2_star

    # Actual accelerations
    x1_2 = u1 + x1_1 + x2_1
    x2_2 = u2

    # Integrate (update state)
    x1 = x1 + x1_1 * dt
    x2 = x2 + x2_1 * dt
    x1_1 = x1_1 + x1_2 * dt
    x2_1 = x2_1 + x2_2 * dt

    # Deviations
    delta1 = (x1_1 ** 2) + (x2_1 ** 2) - v0 ** 2
    delta2 = x1 - np.sin(w * t)

    # Save data
    t_list.append(t)
    delta1_list.append(delta1)
    delta2_list.append(delta2)
    u1_list.append(u1)
    u2_list.append(u2)

    t += dt


# === Plot results ===
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t_list, delta1_list, label="δ₁(t)")
plt.title("Deviation δ₁(t): velocity error")
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_list, delta2_list, label="δ₂(t)")
plt.title("Deviation δ₂(t): x₁ + x₂ ≈ 2")
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_list, u1_list, label="u₁(t)")
plt.title("Control input u₁(t)")
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_list, u2_list, label="u₂(t)")
plt.title("Control input u₂(t)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
