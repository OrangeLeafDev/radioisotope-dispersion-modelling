import numpy as np
import matplotlib.pyplot as plt

metals = {
    "Nickel (Ni)": {
        "v": 0.8, "D": 5.0, "k": 1e-5, "source": 0.0002,
        "runoff_concentration": 0.1
    },
    "Chromium (Cr)": {
        "v": 0.8, "D": 5.0, "k": 1e-5, "source": 0.0002,
        "runoff_concentration": 0.05
    },
}
models_result = {}

# Rainfall-runoff (SCS-CN)
# with cumulative rainfall
# P - rainfall depth
# CN - curve number from land cover / soil group)
def scs_runoff(rainfall, CN):
    cumulative = np.cumsum(rainfall)
    S = (25400 / CN) - 254
    cumulative_runoff = np.where(cumulative > 0.2*S, (cumulative - 0.2*S)**2 / (cumulative + 0.8*S), 0)
    return np.diff(cumulative_runoff, prepend=0)
# Hydraulic parameters (Manning's eq.)
def manning_velocity(n, R, S0):
    return (1/n) * R**(2/3) * S0**0.5
# 1D Advection-Dispersion-Reaction solver (explicit FD)
def adr_model(C0, v, D, k, S_source, dx, dt, nx, nt):
    C = np.zeros((nt, nx))
    C[0] = C0
    for t in range(1, nt):
        prev = C[t-1]
        C[t, 1:-1] = prev[1:-1] + dt * (
            D * (prev[2:] - 2*prev[1:-1] + prev[:-2]) / dx**2
            - v * (prev[1:-1] - prev[:-2]) / dx
            - k * prev[1:-1]
            + S_source[t, 1:-1]
        )
        C[t, 0] = C[t, 1]
        C[t, -1] = C[t, -2]
    return C
# Example monsoon pulse source term
"""
def monsoon_source(nt, nx, source_node, peak_time, magnitude, duration):
    S = np.zeros((nt, nx))
    t_arr = np.arange(nt)
    pulse = magnitude * np.exp(-((t_arr - peak_time)**2) / (2*(duration/2.355)**2))
    S[:, source_node] = pulse
    return S
"""
# Rainfall-driven containment release
def rainfall_source_old(rainfall, CN, nx, source_node, release_factor):
    runoff = scs_runoff(rainfall, CN)
    source = np.zeros((len(rainfall), nx))
    source[:, source_node] = runoff * release_factor
    return source
# Revised function where:
# change of C = (V runoff * C runoff) / V river
def rainfall_source(runoff, nx, source_node, metal_concentration, catchment_area, river_discharge, dt):
    source = np.zeros((len(runoff), nx))
    runoff_volume = runoff * catchment_area
    metal_mass = runoff_volume * metal_concentration
    river_volume = river_discharge * dt * 1000
    source[:, source_node] = metal_mass / river_volume / dt
    return source
# Run example scenario
nx, nt = 200, 500
dx, dt = 50, 10 # meters, seconds
rainfall = np.zeros(nt)    # mm per timestep
rainfall[95:106] = 20 / 11 # Example: 20 mm per timestep during a storm
CN = 80                    # !!!Placeholder curve number
catchment_area = 10_000 # meter squared [placeholder]
river_discharge = 0.5   # meter cubed per second [placeholder]
runoff = scs_runoff(rainfall,CN)
#v, D, k = 0.8, 5.0, 1e-5
for name, m in metals.items():
    C0 = np.zeros(nx)
    source = rainfall_source(
        runoff, nx, source_node=5,
        metal_concentration=m["runoff_concentration"],
        catchment_area=catchment_area,
        river_discharge=river_discharge,
        dt=dt
    )
    models_result[name] = adr_model(
        C0, m["v"], m["D"], m["k"],
        source, dx, dt, nx, nt
    )
    result = models_result[name]
    print(name, "minimum concentration:", result.min())
    print("Location (time, distance):", np.unravel_index(
        np.argmin(result), result.shape
    ))
    print("Minimum source value:", source.min())



fig, axes = plt.subplots(1, len(models_result), figsize=(6*len(models_result), 4))
axes = np.atleast_1d(axes)

for ax, (name, result) in zip(axes, models_result.items()):
    im = ax.imshow(
        result.T, aspect="auto", origin="lower", cmap="inferno",
        extent=[0, nt*dt/3600, 0, nx*dx/1000],
        vmin=0
    )
    ax.set_title(name)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance downstream (km)")
    fig.colorbar(im, ax=ax, label="Concentration (mg/L)")

plt.tight_layout()
plt.show()
"""
plt.imshow(models_result["Nickel (Ni)"].T, aspect='auto', origin='lower', cmap='inferno',
    extent=[0, nt*dt/3600, 0, nx*dx/1000])
plt.xlabel("Time (hours)")
plt.ylabel("Distance downstream (km)")
plt.colorbar(label="Concentration (mg/L)")
plt.title("Simulated Heavy Metal Dispersion During Monsoon Event")
plt.show()
"""