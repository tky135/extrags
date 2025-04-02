import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Initialize parameters
l_init, m_init = 2, 1

# Create spherical coordinate grid
theta, phi = np.meshgrid(np.linspace(0, np.pi, 50), 
                         np.linspace(0, 2*np.pi, 50))

# Set up plot and slider layout
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', facecolor='none')
plt.subplots_adjust(left=0.25, bottom=0.25)

# Create sliders
ax_l = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_m = plt.axes([0.25, 0.05, 0.65, 0.03])
l_slider = Slider(ax=ax_l, label='Orbital (l)', valmin=0, valmax=8, valinit=l_init, valstep=1)
m_slider = Slider(ax=ax_m, label='Magnetic (m)', valmin=-l_init, valmax=l_init, valinit=m_init, valstep=1)

# Initial plot
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
Y = sph_harm(m_init, l_init, phi, theta).real
norm_Y = (Y - Y.min()) / (Y.max() - Y.min())
surf = ax.plot_surface(x, y, z, facecolors=cm.coolwarm(norm_Y), rstride=1, cstride=1, linewidth=0)

# Formatting
ax.set_axis_off()
ax.set_box_aspect([1,1,1])
plt.savefig(f'./spherical_harmonics_{l_init}_{m_init}.png', dpi=300)