import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pennylane as qml

parameters = np.load('/Users/wusixuan/coding/python/Quantum_neuron/Parallelism/non-orthogonal/code/params_ortho.npy')

# Visualize the decision boundary of the trained QNN
beta, theta, phi = np.mgrid[0:np.pi:21j, 
                       0:np.pi:51j, 
                       0:np.pi*2:21j] # Prepare some coordinates
x = beta*np.sin(theta)*np.cos(phi)
y = beta*np.sin(theta)*np.sin(phi)
z = beta*np.cos(theta)

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

betac, thetac, phic = midpoints(beta), midpoints(theta), midpoints(phi)
sphere = np.ones([20,50,20])

# Convert coordinates to amplitudes
alpha_1 = np.sin(betac) * np.cos(thetac)
alpha_2 = np.sin(betac) * np.sin(thetac) * np.cos(phic)
alpha_3 = np.sin(betac) * np.sin(thetac) * np.sin(phic)
alpha_4 = np.cos(betac)

amplitudes = [[a1, a2, a3, a4] for a1,a2,a3,a4 in 
              zip(alpha_1.ravel(), alpha_2.ravel(),
                  alpha_3.ravel(), alpha_4.ravel())]

# Test data
dev = qml.device('default.qubit', wires=3)
# Define the quantum function
@qml.qnode(dev)
def qnn_test(params, state=None):
    # Prepare the initial state
    if state is not None:
        qml.QubitStateVector(state, wires=[0, 1])
    # Apply the rotation gates
    qml.CRY(params[0], wires=[0, 2])
    qml.CRY(params[1], wires=[1, 2])
    qml.RY(params[2], wires=2)

    # Return the expectation value of Z
    return qml.expval(qml.PauliZ(2))

out = np.zeros(len(amplitudes))
for j in range(len(amplitudes)):
    out[j] = qnn_test(parameters, amplitudes[j])
out = out.reshape(20,50,20)

# Choose a color map
cmap = matplotlib.cm.get_cmap("coolwarm")

# Normalize the range for color to -1 - 1
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
colors = cmap(norm(out))

ax = plt.figure().add_subplot(projection='3d')

# Add normalized color bar
m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
m.set_array([])
plt.colorbar(m, shrink=0.7, aspect=20*0.7)

# Modify the sphere variable to dig out the upper hemisphere
upper_half_sphere = (thetac < np.pi / 2) & (phic >= np.pi*4/3)
sphere[upper_half_sphere] = False

# Add voxels to form a solid sphere
voxels = ax.voxels(x, y, z, sphere,
                   cmap=cmap,
                   facecolors=colors,
                   edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                   linewidth=0.1)

ax.set_box_aspect([1, 1, 1])

plt.show()