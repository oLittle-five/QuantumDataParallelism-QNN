# Visualize the decision boundary for the XOR problem solved using non-orthogonal encoding
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pennylane as qml
from matplotlib.colors import ListedColormap

# Define the states
initial_states = [np.array([1, 0]),
                  np.array([np.sqrt(1/3), np.sqrt(2/3)]),
                  np.array([1j*np.sqrt(1/3), 1j*np.sqrt(2/3) * np.exp(1j*4*np.pi/3)]),
                  np.array([np.sqrt(1/3), np.sqrt(2/3) * np.exp(1j*2*np.pi/3)])] 


# Parameters
params = np.load('/Users/wusixuan/coding/python/Quantum_neuron/Parallelism/non-orthogonal/code/params_nonortho.npy')

dev = qml.device('default.qubit', wires=1)
# Define the U3 gate
@qml.qnode(dev)
def qnn_rot(params, state=None):
    # Prepare the initial state
    if state is not None:
        qml.QubitStateVector(state, wires=0)
    # U3 gate on initial state
    qml.Rot(params[0], params[1], params[2], wires=0)
    return qml.state()

# Calculate states
states = np.array([qnn_rot(params, state) for state in initial_states])

for i, state in enumerate(states):
    if state[0]*state[1]==0:
        x_q = 0
        y_q = 0
        if state[0]==0:
            z_q = -1
        else:
            z_q = 1
    else:
        theta_q = 2 * np.arccos(np.linalg.norm(state[0]))
        phi_q = np.real(-1j * np.log((state[1] * np.conjugate(state[0]))
                    /(np.linalg.norm(state[0]) * np.linalg.norm(state[1]))))
        x_q = np.sin(theta_q) * np.cos(phi_q)
        y_q = np.sin(theta_q) * np.sin(phi_q)
        z_q = np.cos(theta_q)
    # Tetrahedron vertices
    globals()[f'state_{i}'] = [x_q, y_q, z_q]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plotting the tetrahedron edges
edges = [(state_0, state_1), (state_0, state_2), 
         (state_0, state_3), (state_1, state_2), 
         (state_1, state_3), (state_2, state_3)]

dashed_edges = [(state_1, state_2), (state_0, state_1), (state_1, state_3)]

for edge in edges:
    if edge in dashed_edges:
        ax.plot3D(*zip(*edge), color="k", linestyle='--')
    else:
        ax.plot3D(*zip(*edge), color="k")

ax.scatter(*state_0, color="red", s=50)
ax.scatter(*state_1, color="blue", s=50)
ax.scatter(*state_2, color="blue", s=50)
ax.scatter(*state_3, color="red", s=50)

# Bloch sphere
theta, phi = np.mgrid[0:np.pi:51j, 0:np.pi*2:51j]
x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

thetac, phic = midpoints(theta), midpoints(phi)

# Convert coordinates to amplitudes
alpha_1 = np.cos(thetac/2)
alpha_2 = np.exp(1j * phic) * np.sin(thetac/2)

amplitudes = [[a1, a2] for a1,a2 in zip(alpha_1.ravel(), alpha_2.ravel())]

# Test data
dev = qml.device('default.qubit', wires=2)
# Define the quantum function
@qml.qnode(dev)
def qnn_test(params, state=None):
    # Prepare the initial state
    if state is not None:
        qml.QubitStateVector(state, wires=0)
    # Apply the rotation gates
    qml.CRY(params[3], wires=[0, 1])
    qml.RY(params[4], wires=1)

    # Return the expectation value of Z
    return qml.expval(qml.PauliZ(1))

out = np.zeros(len(amplitudes))
for j in range(len(amplitudes)):
    out[j] = qnn_test(params, amplitudes[j])

out = out.reshape(50,50)

coolwarm = matplotlib.cm.get_cmap("coolwarm", 256)
newcolors = coolwarm(np.linspace(0, 1, 256))
pink = np.array([0/256, 0/256, 0/256, 1])
newcolors[123:133, :] = pink
newcmp = ListedColormap(newcolors)
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
colors = newcmp(norm(out))

# Add normalized color bar
m = matplotlib.cm.ScalarMappable(cmap=newcmp, norm=norm)
m.set_array([])
plt.colorbar(m, shrink=0.7, aspect=20*0.7, alpha=0.6)

voxels = ax.plot_surface(x, y, z,
                   cmap=newcmp,
                   facecolors=colors,
                   linewidth=0.1,
                   alpha=0.4)

# Determine the limits of your data
x_lim = [0, 0.3]
y_lim = [0, 0.3]
z_lim = [0, 0.3]

def rotation_matrix_rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_ry(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi)],
                     [0, 1, 0],
                     [-np.sin(phi), 0, np.cos(phi)]])

def rzryrz_rotation(vec, theta, phi, psi):
    Rz1 = rotation_matrix_rz(theta)
    Ry = rotation_matrix_ry(phi)
    Rz2 = rotation_matrix_rz(psi)
    return np.dot(Rz2, np.dot(Ry, np.dot(Rz1, vec)))

# Define rotation angles in radians
theta = params[0]
phi = params[1]
psi = params[2]

lines = [
    [x_lim, [y_lim[0], y_lim[0]], [z_lim[0], z_lim[0]]],
    [[x_lim[0], x_lim[0]], y_lim, [z_lim[0], z_lim[0]]],
    [[x_lim[0], x_lim[0]], [y_lim[0], y_lim[0]], z_lim]
]

for index, line in enumerate(lines):
    start_point = np.array([line[0][0], line[1][0], line[2][0]])
    end_point = np.array([line[0][1], line[1][1], line[2][1]])

    linestyle = '--' if index == 0 else '-'
    
    # Rotated vector
    rotated_start = rzryrz_rotation(start_point, theta, phi, psi) - 0.9
    rotated_end = rzryrz_rotation(end_point, theta, phi, psi) - 0.9
    ax.plot([rotated_start[0], rotated_end[0]], 
            [rotated_start[1], rotated_end[1]], 
            [rotated_start[2], rotated_end[2]], linestyle, color='gray')
    ax.quiver(rotated_end[0], rotated_end[1], rotated_end[2], 
              rotated_end[0]-rotated_start[0], rotated_end[1]-rotated_start[1], rotated_end[2]-rotated_start[2], 
              color='gray', length=0.3)

ax.set_box_aspect([1, 1, 1])

plt.show()
