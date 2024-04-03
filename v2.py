import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


vecs = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, -1],
    [1, 1, -1, 1],
    [1, -1, 1, 1],
    [-1, 1, 1, 1],
    [1, 1, -1, -1],
    [1, -1, 1, -1],
    [-1, 1, 1, -1],
    [1, -1, -1, 1],
    [-1, 1, -1, 1],
    [-1, -1, 1, 1],
    [-1, -1, -1, 1],
    [-1, -1, 1, -1],
    [-1, 1, -1, -1],
    [1, -1, -1, -1],
    [-1, -1, -1, -1],
])

connections = []
for i in range(len(vecs)-1):
    for j in range(i, len(vecs)):
        if np.linalg.norm(vecs[i]-vecs[j]) == 2:
            connections.append([i, j])

# vecs = vecs * np.array([0.5, 2, 1.5, 0.75])


A = np.array([2, 2, 2, 2])
p = 0.2



a, b, c, d = 0, 0, 0, 1
n = np.linalg.norm(np.array([a, b, c, d]))

basis = np.array([
    [b, -a, d, -c],
    [c, d, -a, -b],
    [d, c, -b, -a]
])/n

u = basis[1] - (basis[1] @ basis[0])*basis[0]
u = u/np.linalg.norm(u)
v = basis[2] - (basis[2] @ basis[0])*basis[0] - (basis[2] @ u)*u
v = v/np.linalg.norm(v)

o_base = np.array([basis[0], u, v])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect("equal")

data = vecs @ o_base.T
# data = vecs[:, 0:3]
xs = data[:, 0]
ys = data[:, 1]
zs = data[:, 2]
lines = []
for i, j in connections:
    lines.append(ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]])[0])


def update(frame):
    phi = 2*np.pi*frame/100
    R = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), 0, np.sin(phi)],
        [0, 0, 1, 0],
        [0, -np.sin(phi), 0, np.cos(phi)]
    ])
    data = (R @ vecs.T).T @ o_base.T
    # data = (R@vecs.T).T[:, 0:3]
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    for line, ij in zip(lines, connections):
        i, j = ij
        line.set_data(np.array([[xs[i], xs[j]],
                                [ys[i], ys[j]]]))
        line.set_3d_properties([zs[i], zs[j]])


Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800*200)

ani = animation.FuncAnimation(fig=fig, func=update, frames=30*10, interval=100/3)
# ani.save('lines_circle.mp4', writer=writer)
plt.show()
