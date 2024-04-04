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
    [-1, -1, -1, -1]
])

connections = []
vecs2 = []
k = len(vecs)-1
for i in range(len(vecs)-1):
    for j in range(i, len(vecs)):
        if np.linalg.norm(vecs[i]-vecs[j]) == 2:
            k += 1
            connections.append([i, k])
            connections.append([k, j])
            vecs2.append((vecs[i]+vecs[j])/2)

vecs2 = list(np.array(vecs2)*1.2)
print(vecs.shape)
vecs = np.array(list(vecs) + vecs2)
print(vecs.shape)

vecs = vecs * np.array([1, 1, 1, 1])

S = np.array([0, 0, 0, 3])
p = 0.25

n = -S
nm = np.linalg.norm(n)
n = n/nm

a, b, c, d = n

basis = np.array([
    [b, -a, d, -c],
    [c, d, -a, -b],
    [d, c, -b, -a]
]) / nm

u = basis[1] - (basis[1] @ basis[0])*basis[0]
u = u/np.linalg.norm(u)
v = basis[2] - (basis[2] @ basis[0])*basis[0] - (basis[2] @ u)*u
v = v/np.linalg.norm(v)

o_base = np.array([basis[0], u, v])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect("equal")
# ax.set_axis_off()

Bs = (vecs-S).T
n = n.reshape(4, 1)
data = o_base @ (p*(Bs/(n.T @ Bs) - n))

xs = data[0, :]
ys = data[1, :]
zs = data[2, :]
lines = []
for i, j in connections:
    lines.append(ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], marker='o')[0])


def update(frame):
    phi = 2*np.pi*frame/100
    R = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), 0, np.sin(phi)],
        [0, 0, 1, 0],
        [0, -np.sin(phi), 0, np.cos(phi)]
    ])
    rvecs = R @ vecs.T
    Bs = (rvecs.T-S).T
    data = o_base @ (p * (Bs / (n.T @ Bs) - n))

    xs = data[0, :]
    ys = data[1, :]
    zs = data[2, :]
    for line, ij in zip(lines, connections):
        i, j = ij
        line.set_data(np.array([[xs[i], xs[j]],
                                [ys[i], ys[j]]]))
        line.set_3d_properties([zs[i], zs[j]])
    global ax
    # ax.view_init(elev=30, azim=45+phi*180/np.pi/5)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800*200)

ani = animation.FuncAnimation(fig=fig, func=update, frames=30*10, interval=100/3)
# ani.save('projection.mp4', writer=writer)
plt.show()
