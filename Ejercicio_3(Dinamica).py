import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# gravedad
g = 9.81

# masas
m1 = 4
m2 = 2
m3 = 1

# aceleraciones del sistema
a2 = ((2*m1 - m2 - m3)*g) / (4*m1 + m2 + m3)
a1 = 2*a2

# tiempo de simulación
dt = 0.05
t = np.arange(0, 5, dt)

# posiciones (MRUA)
x1 = 0.5 * a1 * t**2
x2 = 0.5 * a2 * t**2
x3 = x2

# figura
fig, ax = plt.subplots()

ax.set_xlim(-2,2)
ax.set_ylim(-6,2)
ax.set_aspect('equal')
ax.set_title("Simulación sistema de poleas")

# polea fija
top_pulley = plt.Circle((0,1),0.2,fill=False,lw=2)

# polea móvil
mov_pulley = plt.Circle((0,-1),0.2,fill=False,lw=2)

ax.add_patch(top_pulley)
ax.add_patch(mov_pulley)

# bloques
block1, = ax.plot([],[], 's', markersize=12)
block2, = ax.plot([],[], 's', markersize=12)
block3, = ax.plot([],[], 's', markersize=12)

# cuerda
rope, = ax.plot([],[], lw=2)

def update(frame):

    # posiciones verticales
    y1 = -x1[frame]
    y2 = -1 - x2[frame]
    y3 = y2

    # actualizar bloques
    block1.set_data([-1], [y1])
    block2.set_data([0.4], [y2])
    block3.set_data([-0.4], [y3])

    # mover polea móvil
    mov_pulley.center = (0, -1 - x2[frame])

    # cuerda simplificada
    xs = [-1, 0, 0, 0.4]
    ys = [y1, 1, -1 - x2[frame], y2]

    rope.set_data(xs, ys)

    return block1, block2, block3, rope, mov_pulley

# animación
ani = FuncAnimation(fig, update, frames=len(t), interval=50)

plt.grid()
plt.show()