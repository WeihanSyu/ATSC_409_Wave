import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate


class Wave_field():
    def __init__(self, n, m, l):
        self.n = n
        self.m = m
        self.l = l
        self.prev = np.empty((m+1, n+1))
        self.now = np.empty((m+1, n+1))
        self.next = np.empty((m+1, n+1))
        self.field = np.empty((l+1, m+1, n+1))

    def fullgrid_time(self, timestep, attr='next'):
        self.field[timestep, :, :] = self.__getattribute__(attr)

    def shift(self):
        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)


def initial_cond(u, v, h, t, ho):
    u.prev[:, :] = 0
    v.prev[:, :] = 0
    h.prev[:, :] = 0
    h.prev[(len(h.prev)*3//4), (len(h.prev[0, :])//5)] = ho*np.cos(np.pi*t/0.024)
    h.prev[(len(h.prev)//2), (len(h.prev[0, :])*4//5)] = ho*np.cos(np.pi*t/0.024)


# Note: For our boundary conditions, we are using 'no-slip' and 'no-penetration' conditions so fluid particles on
# the wall move with the velocity of the wall (zero in this case)
def b_cond(u_array, v_array, h_array, n, m, dt, ho, timestep):
    t = dt*timestep

    u_array[:, 0] = 0
    u_array[:, n] = 0
    u_array[0, :] = 0
    u_array[m, :] = 0

    v_array[0, :] = 0
    v_array[m, :] = 0
    v_array[:, 0] = 0
    v_array[:, n] = 0

    h_array[:, 0] = 0
    h_array[:, n] = 0
    h_array[0, :] = 0
    h_array[m, :] = 0

    h_array[(len(h_array) * 3 // 4), (len(h_array[0, :]) // 5)] = ho * np.cos(np.pi * t / 0.024)
    h_array[(len(h_array) // 2), (len(h_array[0, :]) * 4 // 5)] = ho * np.cos(np.pi * t / 0.024)


def t1(u, v, h, g, H, dx, dy, dt, ho):
    u.now[:, :] = 0
    u.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 - 1)] = -dt * g * ho / dx
    u.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 + 1)] = dt * g * ho / dx
    u.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 - 1)] = -dt * g * ho / dx
    u.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 + 1)] = dt * g * ho / dx

    v.now[:, :] = 0
    v.now[(len(u.now) * 3 // 4 - 1), (len(u.now[0, :]) // 5)] = -dt * g * ho / dy
    v.now[(len(u.now) * 3 // 4 + 1), (len(u.now[0, :]) // 5)] = dt * g * ho / dy
    v.now[(len(u.now) // 2 - 1), (len(u.now[0, :]) * 4 // 5)] = -dt * g * ho / dy
    v.now[(len(u.now) // 2 + 1), (len(u.now[0, :]) * 4 // 5)] = dt * g * ho / dy

    h.now[:, :] = 0
    A = dt ** 2 * g * ho
    h.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5)] = ho - (A / 2 / dx ** 2) * (ho + H) - (A / 2 / dy ** 2) * (
                ho + H)
    h.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5)] = ho - (A / 2 / dx ** 2) * (ho + H) - (A / 2 / dy ** 2) * (
                ho + H)

    h.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 - 1)] = A * ho / 4 / dx ** 2
    h.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 + 1)] = A * ho / 4 / dx ** 2
    h.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 - 1)] = A * ho / 4 / dx ** 2
    h.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 + 1)] = A * ho / 4 / dx ** 2

    h.now[(len(u.now) * 3 // 4 - 1), (len(u.now[0, :]) // 5)] = A * ho / 4 / dy ** 2
    h.now[(len(u.now) * 3 // 4 + 1), (len(u.now[0, :]) // 5)] = A * ho / 4 / dy ** 2
    h.now[(len(u.now) // 2 - 1), (len(u.now[0, :]) * 4 // 5)] = A * ho / 4 / dy ** 2
    h.now[(len(u.now) // 2 + 1), (len(u.now[0, :]) * 4 // 5)] = A * ho / 4 / dy ** 2

    h.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 - 2)] = A * H / 4 / dx ** 2
    h.now[(len(u.now) * 3 // 4), (len(u.now[0, :]) // 5 + 2)] = A * H / 4 / dx ** 2
    h.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 - 2)] = A * H / 4 / dx ** 2
    h.now[(len(u.now) // 2), (len(u.now[0, :]) * 4 // 5 + 2)] = A * H / 4 / dx ** 2

    h.now[(len(u.now) * 3 // 4 - 2), (len(u.now[0, :]) // 5)] = A * H / 4 / dy ** 2
    h.now[(len(u.now) * 3 // 4 + 2), (len(u.now[0, :]) // 5)] = A * H / 4 / dy ** 2
    h.now[(len(u.now) // 2 - 2), (len(u.now[0, :]) * 4 // 5)] = A * H / 4 / dy ** 2
    h.now[(len(u.now) // 2 + 2), (len(u.now[0, :]) * 4 // 5)] = A * H / 4 / dy ** 2


# About the leap_frog function below:
# First of all our arrays have m+1 rows and n+1 columns thus, the indices are from 0:m and 0:n
# Recall in Python, index calling works like range() function where [1:m] calls [1:m-1] index wise
# Our boundary conditions already take care of [:, 0], [:, n], [0, :], [m, :] so we need to exclude these areas.
def leap_frog(u, v, h, g, H, m, n, dt, dx, dy):
    for i in np.arange(1, n):
        u.next[1:m, i] = u.prev[1:m, i] - dt*u.now[1:m, i]*(u.now[1:m, i+1] - u.now[1:m, i-1])/dx - \
                         dt*v.now[1:m, i]*(u.now[(1+1):(m+1), i] - u.now[(1-1):(m-1), i])/dy - \
                         dt*g*(h.now[1:m, i+1] - h.now[1:m, i-1])/dx

        v.next[1:m, i] = v.prev[1:m, i] - dt*u.now[1:m, i]*(v.now[1:m, i+1] - v.now[1:m, i-1])/dx - \
                         dt*v.now[1:m, i]*(v.now[(1+1):(m+1), i] - v.now[(1-1):(m-1), i])/dy - \
                         dt*g*(h.now[(1+1):(m+1), i] - h.now[(1-1):(m-1), i])/dy

        h.next[1:m, i] = h.prev[1:m, i] - dt*u.now[1:m, i]*(h.now[1:m, i+1] - h.now[1:m, i-1])/dx - \
                         dt*(h.now[1:m, i] + H)*(u.now[1:m, i+1] - u.now[1:m, i-1])/dx - \
                         dt*v.now[1:m, i]*(h.now[(1+1):(m+1), i] - h.now[(1-1):(m-1), i])/dy - \
                         dt*(h.now[1:m, i] + H)*(v.now[(1+1):(m+1), i] - v.now[(1-1):(m-1), i])/dy


def ball_water(u, v, p, mb, vbi, dx, dy, n, m, l, dt):
    r = 0.5  # multiplier for our momentum equations
    mw = 4 / 3 * np.pi * (dx / 2) ** 3 * p  # mass of water

    Pixw = mw * u.field
    Pfxw = r * Pixw
    Pixb = np.zeros(l + 1)  # np.zeros makes floats, we need an integer value first to start off our indices.
    Pixb[0] = 0  # so manually input 0
    Pfxb = np.zeros(l + 1)
    xstep = np.zeros(l + 1)
    xstep[0] = int(round((n + 1) / 2))  # round() may make floats, so again use int() to convert.
    Pfxb[0] = Pixw[0, 0, int(xstep[0])] + Pixb[0] - Pfxw[0, 0, int(xstep[0])]
    vfxb = np.zeros(l + 2)
    vfxb[1] = Pfxb[0] / mb
    xspace = np.zeros(l + 1)
    xspace[0] = (dx * n + dx) / 2

    Piyw = mw * v.field
    Pfyw = r * Piyw
    Piyb = np.zeros(l + 1)
    Piyb[0] = mb * vbi
    Pfyb = np.zeros(l + 1)
    Pfyb[0] = Piyw[0, 0, int(xstep[0])] + Piyb[0] - Pfyw[0, 0, int(xstep[0])]
    vfyb = np.zeros(l + 2)
    vfyb[0] = vbi
    vfyb[1] = Pfyb[0] / mb
    yspace = np.zeros(l + 1)
    ystep = np.zeros(l + 1)
    ystep[0] = 0

    for i in np.arange(1, l + 1):
        xspace[i] = vfxb[i] * dt + xspace[i - 1]
        if xspace[i] < 0:
            xspace[i] = 0
        elif xspace[i] > (dx * n):
            xspace[i] = (dx * n)
        yspace[i] = vfyb[i] * dt + yspace[i - 1]
        if yspace[i] < 0:
            yspace[i] = 0
        elif yspace[i] > (dy * m):
            yspace[i] = (dy * m)

        xstep[i] = round(xspace[i] / dx)
        ystep[i] = round(yspace[i] / dy)

        Pixb[i] = mb * vfxb[i]
        Pfxb[i] = Pixw[i, int(ystep[i]), int(xstep[i])] + Pixb[i] - Pfxw[i, int(ystep[i]), int(xstep[i])]
        vfxb[i + 1] = Pfxb[i] / mb
        Piyb[i] = mb * vfyb[i]
        Pfyb[i] = Piyw[i, int(ystep[i]), int(xstep[i])] + Piyb[i] - Pfyw[i, int(ystep[i]), int(xstep[i])]
        vfyb[i + 1] = Pfyb[i] / mb

    return [xspace, yspace]


def model(args):
    l = int(args[0])
    m = int(args[1])
    n = int(args[2])

    g = 980  # gravitational accel [cm/s^2]
    H = 10  # [cm]
    dt = 0.001  # [s]
    dx = 0.3  # [cm]
    dy = 0.3  # [cm]
    ho = 4  # initial source perturbation height [cm]
    p = 1E-3  # density water [kg/cm^3]
    mb = 1E-6  # mass ball [kg]
    vbi = 5  # initial velocity ball [cm/s]

    u = Wave_field(n, m, l)
    v = Wave_field(n, m, l)
    h = Wave_field(n, m, l)

    # Changes the .prev wave_fields to have values from initial_cond function
    initial_cond(u, v, h, 0, ho)
    # Store values at specified time to our full grid.
    u.fullgrid_time(0, 'prev')
    v.fullgrid_time(0, 'prev')
    h.fullgrid_time(0, 'prev')

    t1(u, v, h, g, H, dx, dy, dt, ho)
    # Note our boundary conditions require a certain x and y size or else we will be out of bounds
    b_cond(u.now, v.now, h.now, n, m, dt, ho, 1)
    u.fullgrid_time(1, 'now')
    v.fullgrid_time(1, 'now')
    h.fullgrid_time(1, 'now')

    for timestep in np.arange(2, l + 1):
        leap_frog(u, v, h, g, H, m, n, dt, dx, dy)
        b_cond(u.next, v.next, h.next, n, m, dt, ho, timestep)
        u.fullgrid_time(timestep)
        v.fullgrid_time(timestep)
        h.fullgrid_time(timestep)
        u.shift()
        v.shift()
        h.shift()

    # Call our momentum equations
    [xspace, yspace] = ball_water(u, v, p, mb, vbi, dx, dy, n, m, l, dt)

    # Create the ball
    b = np.linspace(0, 2 * np.pi, 100)
    c = np.linspace(0, np.pi, 100)
    x = dx / 2 * np.outer(np.cos(b), np.sin(c))
    y = dy / 2 * np.outer(np.sin(b), np.sin(c))
    z = dx / 2 * np.outer(np.ones(np.size(b)), np.cos(c))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(0, dx * n)
    ax.set_ylim(0, dy * m)
    ax.set_zlim(-H, 7)
    '''X = np.arange(0, (dx * n + dx), dx)     # works like range()
    Y = np.arange(0, (dy * m + dy), dy)
    X, Y = np.meshgrid(X, Y)'''

    # Plot every second grid point instead:
    X = np.arange(0, (dx * n + dx), 2*dx)
    Y = np.arange(0, (dy * m + dy), 2*dy)
    X, Y = np.meshgrid(X, Y)

    def animation_frame(i):
        # h.field[i, 0:(m+1):2, 0:(n+1):2]
        # h.field[i, :, :]
        ax.clear()
        ax.set_xlim(0, dx * n)
        ax.set_ylim(0, dy * m)
        ax.set_zlim(-H, 7)
        # ax.grid(False)
        # surf = ax.plot_surface(X, Y, h.field[i, :, :], alpha=0.4, cmap='Blues', linewidth=0, antialiased=False)
        surf = ax.plot_surface(X, Y, h.field[i, 0:(m+1):2, 0:(n+1):2], alpha=0.4, cmap='Blues', linewidth=0, antialiased=False)
        # surf = ax.plot_surface(x + xspace[i], y + yspace[i], z - (dx / 2), color='black')
        return
    
    animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, (l + 1), 1), interval=5, repeat=True)

    plt.show()
    return 

# The ups and downs are not instability.  Because I have not staggered my grid it's only using every
# second grid point.  Plotting every second grid point should give a nice smooth flow.

# IMPORTANT NOTE: When plotting every second point, we are getting even numbered indices only and sometimes this
# may exclude the points that our two drums lie on. Thus we must always choose m and n such that the drum points are
# always included.
# drum 1: [(m+1)*3//4, (n+1)//5]
# drum 2: [(m+1)//2, (n+1)*4//5]
# These must all be even numbers. m = 47, n = 20 works

# The reason our animation moves so choppy is because the animation has to follow the slower one which is our object.
# The ball just moves choppy.
# To rectify this, we would need to give more frames for our ball but I don't know how to combine this with the waves
model((150,47,20))