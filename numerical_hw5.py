import numpy as np
import math
import pandas as pd

# 第1題

def f(t, y):
    return 1 + (y/t) + (y/t)**2

def df(t, y):
    partial_f_t = -(y / t**2) - 2*(y**2) / (t**3)
    partial_f_y = (1/t) + (2*y) / (t**2)
    return partial_f_t + partial_f_y * f(t, y)

def exact_solution(t):
    return t * math.tan(math.log(t))

t0 = 1.0
y0 = 0.0
h = 0.1
n_steps = int((2.0 - t0)/h)

t_values = [t0]
y_euler = [y0]
y_taylor2 = [y0]
y_exact = [exact_solution(t0)]

for i in range(n_steps):
    t = t_values[-1]
    y_e = y_euler[-1]
    y_t = y_taylor2[-1]

    y_e_next = y_e + h * f(t, y_e)
    y_t_next = y_t + h * f(t, y_t) + (h**2 / 2) * df(t, y_t)

    t_next = t + h

    t_values.append(t_next)
    y_euler.append(y_e_next)
    y_taylor2.append(y_t_next)
    y_exact.append(exact_solution(t_next))

data1 = {
    't': t_values,
    'Exact Value': y_exact,
    'Euler Value': y_euler,
    'Euler Relative Error': [abs(e - ex)/abs(ex) if ex != 0 else 0 for e, ex in zip(y_euler, y_exact)],
    'Taylor 2nd Value': y_taylor2,
    'Taylor 2nd Relative Error': [abs(t - ex)/abs(ex) if ex != 0 else 0 for t, ex in zip(y_taylor2, y_exact)]
}

df1 = pd.DataFrame(data1)

pd.set_option('display.float_format', '{:.6f}'.format)
print("第1題")
print(df1)

# 第2題

def F(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*math.cos(t) - (1/3)*math.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*math.cos(t) + (1/3)*math.sin(t)
    return np.array([du1_dt, du2_dt])

def u1_exact(t):
    return 2*math.exp(-3*t) - math.exp(-39*t) + (1/3)*math.cos(t)

def u2_exact(t):
    return -math.exp(-3*t) + 2*math.exp(-39*t) - (1/3)*math.cos(t)

def runge_kutta_system(t0, u0, h, t_end):
    t_values = [t0]
    u1_values = [u0[0]]
    u2_values = [u0[1]]
    u1_exacts = [u1_exact(t0)]
    u2_exacts = [u2_exact(t0)]
    errors_u1 = [abs(u0[0] - u1_exact(t0))/abs(u1_exact(t0)) if u1_exact(t0) != 0 else 0]
    errors_u2 = [abs(u0[1] - u2_exact(t0))/abs(u2_exact(t0)) if u2_exact(t0) != 0 else 0]

    t = t0
    u = np.array(u0)

    while t < t_end:
        k1 = F(t, u)
        k2 = F(t + h/2, u + h/2 * k1)
        k3 = F(t + h/2, u + h/2 * k2)
        k4 = F(t + h, u + h * k3)

        u = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_values.append(t)
        u1_values.append(u[0])
        u2_values.append(u[1])
        u1_exacts.append(u1_exact(t))
        u2_exacts.append(u2_exact(t))
        errors_u1.append(abs(u[0] - u1_exact(t))/abs(u1_exact(t)) if u1_exact(t) != 0 else 0)
        errors_u2.append(abs(u[1] - u2_exact(t))/abs(u2_exact(t)) if u2_exact(t) != 0 else 0)

    df = pd.DataFrame({
        't': t_values,
        'u1 (Exact)': u1_exacts,
        'u1 (RK4)': u1_values,
        'u1 Relative Error': errors_u1,
        'u2 (Exact)': u2_exacts,
        'u2 (RK4)': u2_values,
        'u2 Relative Error': errors_u2
    })

    return df



t0 = 0.0
u0 = [4/3, 2/3]
t_end = 1.0
h_1 = 0.1
h_2 = 0.05

df2_h1 = runge_kutta_system(t0, u0, h_1, t_end)
df2_h2 = runge_kutta_system(t0, u0, h_2, t_end)

print("\n第2題 h=0.1")
print(df2_h1)
print("\n第2題 h=0.05")
print(df2_h2)


#####