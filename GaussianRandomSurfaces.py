import numpy as np
import matplotlib.pyplot as plt

# Function to generate surface roughness
def generate_surface_roughness(x_span, y_span, z_span, sigma_rms, corr_length_x, corr_length_y, delta, 
                                coating, make_plots, seed, index, material, extra_width, detail):
    np.random.seed(seed)
    
    # Calculate the number of points in x and y
    Nx = int(round(x_span / delta)) + 1
    Ny = int(round(y_span / delta)) + 1

    # Generate x and y coordinates
    x = np.linspace(-x_span / 2, x_span / 2, Nx)
    y = np.linspace(-y_span / 2, y_span / 2, Ny)

    # Calculate k vectors
    kx = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    ky = np.fft.fftfreq(len(y), d=(y[1] - y[0]))
    Kx, Ky = np.meshgrid(kx, ky)

    # Create randomized surface in k space
    Zk = np.random.rand(len(kx), len(ky)) * np.exp(1j * 2 * np.pi * np.random.rand(len(kx), len(ky)))

    # Apply correlation length, Gaussian shape in k space
    Zk *= np.exp(-0.125 * ((Kx * corr_length_x) ** 2 + (Ky * corr_length_y) ** 2))

    # Convert to real space
    Zrand = np.fft.ifft2(Zk).real
    Zrand -= np.mean(Zrand)
    Zrand /= np.sqrt(np.mean(Zrand**2))  # Normalize RMS to 1
    Zrand *= sigma_rms  # Scale to desired RMS

    return x, y, Zrand

# Function to analyze subsections
def analyze_subsections_correct(surface, x, y, x_subsections=10, y_subsections=10):
    Nx, Ny = surface.shape
    x_step = Nx // x_subsections
    y_step = Ny // y_subsections

    rms_matrix = np.zeros((x_subsections, y_subsections))

    for i in range(x_subsections):
        for j in range(y_subsections):
            x_start, x_end = i * x_step, (i + 1) * x_step
            y_start, y_end = j * y_step, (j + 1) * y_step

            subsection = surface[x_start:x_end, y_start:y_end]

            # Compute RMS
            rms_matrix[i, j] = np.sqrt(np.mean(subsection**2)) * 1e9  # Convert to nm

    return rms_matrix

# Parameters for surface generation
x_span = 1e-6  # 1 micron
y_span = 1e-6  # 1 micron
z_span = 3e-6  # 3 microns
sigma_rms = 3250e-9  # 100 nm
corr_length_x = 100e-9  # 100 nm
corr_length_y = 300e-9  # 100 nm
delta = 5e-9  # 10 nm
seed = 42

# Generate the surface
x, y, Zrand = generate_surface_roughness(
    x_span, y_span, z_span, sigma_rms, corr_length_x, corr_length_y, delta, 
    0, 0, seed, 1.5, "material", 0, 0.5
)

# Analyze the surface
rms_matrix = analyze_subsections_correct(Zrand, x, y, x_subsections=10, y_subsections=10)

# Extract surface profiles
y_index = np.argmin(np.abs(y))  # Closest index to y=0
x_index = np.argmin(np.abs(x))  # Closest index to x=0
surface_profile_x = Zrand[:, y_index]  # Profile along x (at y=0)
surface_profile_y = Zrand[x_index, :]  # Profile along y (at x=0)

# Create the combined subplot layout with requested shapes: a square for "a" and horizontal rectangles for others
fig = plt.figure(figsize=(16, 10))

# Left large plot: Surface height map (square shape)
ax1 = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=2)
im = ax1.imshow(Zrand * 1e9, extent=(x[0] * 1e9, x[-1] * 1e9, y[0] * 1e9, y[-1] * 1e9), cmap='viridis')
ax1.set_xlabel("X (nm)")
ax1.set_ylabel("Y (nm)")
fig.colorbar(im, ax=ax1, label="Height (nm)")
ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Top right plot: RMS roughness matrix (horizontal rectangle)
ax2 = plt.subplot2grid((3, 5), (0, 2), colspan=3)
im2 = ax2.imshow(rms_matrix, cmap='viridis', origin='lower', aspect='auto')
fig.colorbar(im2, ax=ax2, label='RMS (nm)')
ax2.set_xlabel("Subsection X")
ax2.set_ylabel("Subsection Y")
ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Middle right plot: Surface roughness profile along x (horizontal rectangle)
ax3 = plt.subplot2grid((3, 5), (1, 2), colspan=3)
ax3.plot(x * 1e9, surface_profile_x * 1e9, label="y = 0")
ax3.set_xlabel("X (nm)")
ax3.set_ylabel("Height (nm)")
ax3.grid()
ax3.legend()
ax3.text(-0.1, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Bottom right plot: Surface roughness profile along y (horizontal rectangle)
ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=3)
ax4.plot(y * 1e9, surface_profile_y * 1e9, color='orange', label="x = 0")  # Corrected profile
ax4.set_xlabel("Y (nm)")
ax4.set_ylabel("Height (nm)")
ax4.grid()
ax4.legend()
ax4.text(-0.1, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.show()
