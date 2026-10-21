import os
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator

# Ensure output directory exists
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional GIS & Basemap libraries
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    import contextily as cx
    HAS_GIS = True
except ImportError:
    HAS_GIS = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def get_time_in_seconds(value, unit):
    """Converts flexible time inputs (days, hours, minutes, seconds) into total seconds."""
    unit = str(unit).strip().lower()
    if unit in ["day", "days", "d"]:
        return float(value) * 24.0 * 3600.0
    elif unit in ["hour", "hours", "h", "hr", "hrs"]:
        return float(value) * 3600.0
    elif unit in ["minute", "minutes", "m", "min", "mins"]:
        return float(value) * 60.0
    elif unit in ["second", "seconds", "s", "sec", "secs"]:
        return float(value)
    else:
        raise ValueError("Invalid time unit. Use 'days', 'hours', 'minutes', or 'seconds'.")


def fetch_real_river_and_snap_stations(stations_df):
    """
    Downloads the Bayto / Cabaluan / Santa Cruz River network from OpenStreetMap
    and snaps stations to the active channel. Gracefully falls back if offline.
    """
    gdf_rivers = None
    if HAS_GIS:
        try:
            bbox = (119.885, 15.685, 120.040, 15.745)
            tags = {"waterway": ["river", "stream"]}
            print("Downloading river network from OpenStreetMap for Santa Cruz, Zambales...")
            gdf_rivers = ox.features_from_bbox(bbox=bbox, tags=tags)
            gdf_rivers = gdf_rivers[gdf_rivers.geometry.type.isin(["LineString", "MultiLineString"])]

            if hasattr(gdf_rivers.geometry, "union_all"):
                river_union = gdf_rivers.geometry.union_all()
            else:
                river_union = gdf_rivers.geometry.unary_union

            snapped_lons, snapped_lats = [], []
            for _, row in stations_df.iterrows():
                pt = Point(row["longitude"], row["latitude"])
                nearest_on_water = nearest_points(pt, river_union)[1]
                snapped_lons.append(nearest_on_water.x)
                snapped_lats.append(nearest_on_water.y)

            stations_df = stations_df.copy()
            stations_df["longitude"] = snapped_lons
            stations_df["latitude"] = snapped_lats
            return stations_df, gdf_rivers
        except Exception as e:
            print(f"Notice: OSM fetch skipped ({e}). Using verified station coordinates.")

    return stations_df, gdf_rivers


def get_real_elevation_grid(LON, LAT, r_lon, r_lat, dem_tif_path="srtm_santa_cruz.tif"):
    """
    1. Loads local SRTM DEM raster (.tif) if present.
    2. Queries Open-Meteo SRTM API if online.
    3. Falls back to correct East-to-West Zambales topography if offline.
    """
    if HAS_RASTERIO and os.path.exists(dem_tif_path):
        try:
            print(f"Loading local DEM contours from {dem_tif_path}...")
            with rasterio.open(dem_tif_path) as src:
                coords = [(lon, lat) for lon, lat in zip(LON.ravel(), LAT.ravel())]
                elev = np.array([val[0] for val in src.sample(coords)]).reshape(LON.shape)
                return np.where(elev < -100, 0.0, elev)
        except Exception as e:
            print(f"Failed to read local DEM file: {e}. Falling back to API/model.")

    try:
        print("Fetching SRTM elevation grid from Open-Meteo DEM API...")
        sub_lon = np.linspace(LON.min(), LON.max(), 20)
        sub_lat = np.linspace(LAT.min(), LAT.max(), 16)
        SLON, SLAT = np.meshgrid(sub_lon, sub_lat)

        elev_pts = []
        flat_lats = SLAT.ravel()
        flat_lons = SLON.ravel()
        for i in range(0, len(flat_lats), 80):
            lats_str = ",".join(f"{v:.5f}" for v in flat_lats[i:i+80])
            lons_str = ",".join(f"{v:.5f}" for v in flat_lons[i:i+80])
            url = f"https://api.open-meteo.com/v1/elevation?latitude={lats_str}&longitude={lons_str}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            elev_pts.extend(resp.json()["elevation"])

        rbf_dem = RBFInterpolator(
            np.column_stack([flat_lons, flat_lats]),
            np.array(elev_pts),
            kernel="thin_plate_spline",
            smoothing=1e-3
        )
        return np.maximum(0.0, rbf_dem(np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape))
    except Exception as e:
        print(f"Notice: Elevation API unavailable ({e}). Using analytical Zambales Ophiolite terrain.")

    # Corrected Topography: High elevations in the East (120.03°E ~ 420m), sloping down to Coast in the West (119.89°E ~ 0m)
    east_origin = 120.035
    inland_dist = np.maximum(0.0, east_origin - LON)
    dist_to_river = np.min(
        np.hypot(LON[:, :, None] - r_lon[None, None, :], LAT[:, :, None] - r_lat[None, None, :]),
        axis=2
    )
    base_elev = np.where(LON < east_origin, 420.0 * (inland_dist / 0.14) ** 1.35, 0.0)
    valley_carve = 1.0 - 0.52 * np.exp(-(dist_to_river / 0.012) ** 2)
    ridge_detail = 22.0 * np.sin((LON - 119.9) * 95) * np.cos((LAT - 15.70) * 110) * np.clip(inland_dist / 0.04, 0, 1)
    return np.where(LON < east_origin, np.maximum(1.0, base_elev * valley_carve + ridge_detail), 0.0)


def simulate_n_duration_dispersion(
    total_seconds, nx=200, dx=50.0,
    manning_n=0.040, W_width=25.0, H_depth=1.6, S0_slope=0.0025,
    Kd_metal=1500.0, ssc_mg_L=450.0, v_settle=1.5e-4, bg_sediment_mg_kg=18.0
):
    """
    Runs 1-D ADR + Two-Phase Partitioning & Bed Sediment Accumulation model
    for any arbitrary time duration.
    """
    R_hyd = (W_width * H_depth) / (W_width + 2.0 * H_depth)
    v_flow = (1.0 / manning_n) * (R_hyd ** (2.0 / 3.0)) * (S0_slope ** 0.5)
    u_star = np.sqrt(9.81 * H_depth * S0_slope)
    D_disp = 0.011 * (v_flow ** 2) * (W_width ** 2) / (H_depth * u_star)

    # Automatically enforce Courant stability (Cr + 2*Diff <= 0.8)
    dt_max_adv = 0.5 * dx / max(v_flow, 1e-6)
    dt_max_diff = 0.2 * (dx ** 2) / max(D_disp, 1e-6)
    dt = min(20.0, dt_max_adv, dt_max_diff)
    cr = v_flow * dt / dx
    assert cr <= 1.0, f"Unstable Courant number: {cr}"

    nt = max(2, int(total_seconds / dt))
    ssc_kg_L = ssc_mg_L * 1e-6
    f_d = 1.0 / (1.0 + Kd_metal * ssc_kg_L)
    f_p = 1.0 - f_d
    k_settling = f_p * (v_settle / H_depth)

    C_water = np.full(nx, 0.012)
    Bed_Sediment = np.full(nx, bg_sediment_mg_kg)

    storm_start = int(nt * 0.10)
    storm_end = min(nt, storm_start + max(1, int(7200.0 / dt)))
    source_node = 25  # Inland mining source node (East)
    bed_conversion = 0.085

    for t in range(1, nt):
        prev_C = C_water.copy()
        d2Cdx2 = (prev_C[2:] - 2.0 * prev_C[1:-1] + prev_C[:-2]) / (dx ** 2)
        dCdx = (prev_C[1:-1] - prev_C[:-2]) / dx

        C_water[1:-1] = (
            prev_C[1:-1]
            + dt * (D_disp * d2Cdx2 - v_flow * dCdx - k_settling * prev_C[1:-1])
        )
        runoff_pulse = (12.0 * 0.0025) if (storm_start <= t < storm_end) else 0.0
        C_water[source_node] += (0.0004 + runoff_pulse) * dt

        C_water[0] = 0.012  # Upstream headwater boundary
        C_water[-1] = C_water[-2]
        Bed_Sediment += (k_settling * C_water * dt * Kd_metal) * bed_conversion

    return {
        "C_water_final": C_water,
        "Bed_Sediment_final": Bed_Sediment,
        "dt": dt,
        "nt": nt,
        "v_flow": v_flow,
        "D_disp": D_disp
    }


def generate_predictive_map(
    time_value=14, time_unit="days", analyte_name="Nickel (Ni)",
    dem_tif_path="srtm_santa_cruz.tif", show_3d=True
):
    """
    Unified 2D + True 3D + Pseudo-3D Elevation-Tiered Spatial Interpolation Pipeline.
    Flow goes correctly from Inland Highlands (East) to Coast (West).
    """
    total_seconds = get_time_in_seconds(time_value, time_unit)
    print(f"=== Running Unified Predictive Model for {time_value} {time_unit} ({total_seconds:.0f} s) ===")

    # 1. Stations ordered correctly: Node 0 = Inland East (Source/Control), Node 199 = Coastal West (Outlet)
    stations_df = pd.DataFrame([
        {"location": "Station 00 (Control)",   "longitude": 120.025000, "latitude": 15.699722, "node_idx": 0},
        {"location": "Source (Laterite Mine)", "longitude": 120.000000, "latitude": 15.699444, "node_idx": 25},
        {"location": "Near-Field",             "longitude": 119.989000, "latitude": 15.704200, "node_idx": 60},
        {"location": "Mid-Field",              "longitude": 119.964500, "latitude": 15.713500, "node_idx": 110},
        {"location": "Lower Reach (15.719°N)", "longitude": 119.937874, "latitude": 15.719390, "node_idx": 150},
        {"location": "Outlet (River Mouth)",   "longitude": 119.892000, "latitude": 15.714300, "node_idx": 199},
    ])

    stations_df, gdf_rivers = fetch_real_river_and_snap_stations(stations_df)

    # 2. Interpolate 200 River Nodes from East (Inland) to West (Coast)
    nx = 200
    pts_wp = stations_df[["longitude", "latitude"]].values
    t_wp = np.linspace(0, 1, len(pts_wp))
    t_fine = np.linspace(0, 1, nx)
    r_lon = np.interp(t_fine, t_wp, pts_wp[:, 0])
    r_lat = np.interp(t_fine, t_wp, pts_wp[:, 1])

    # 3. Run Simulation
    sim_out = simulate_n_duration_dispersion(total_seconds=total_seconds, nx=nx, dx=50.0)
    final_bed = sim_out["Bed_Sediment_final"]
    stations_df["predicted_sed_mg_kg"] = final_bed[stations_df["node_idx"].values]

    # 4. Build 2D/3D Topographic Grid
    grid_lon = np.linspace(119.885, 120.035, 160)
    grid_lat = np.linspace(15.685, 15.740, 120)
    LON, LAT = np.meshgrid(grid_lon, grid_lat)

    ELEV = get_real_elevation_grid(LON, LAT, r_lon, r_lat, dem_tif_path=dem_tif_path)

    rbf_elev_river = RBFInterpolator(
        np.column_stack([LON.ravel(), LAT.ravel()])[::8],
        ELEV.ravel()[::8],
        kernel="linear"
    )
    r_elev = np.maximum(1.0, rbf_elev_river(np.column_stack([r_lon, r_lat])))

    rbf_plume = RBFInterpolator(np.column_stack([r_lon, r_lat]), final_bed, kernel="thin_plate_spline", smoothing=0.5)
    z_interp = rbf_plume(np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)

    dist_to_river = np.min(
        np.hypot(LON[:, :, None] - r_lon[None, None, :], LAT[:, :, None] - r_lat[None, None, :]),
        axis=2
    )
    corridor_weight = np.exp(-(dist_to_river / 0.012) ** 1.5)
    bg_val = float(final_bed.min())
    vmax_val = max(float(final_bed.max()), bg_val + 1.0)
    C_GRID = np.where(
        ELEV > 0.5,
        np.clip(z_interp * corridor_weight + bg_val * (1.0 - corridor_weight), bg_val, vmax_val),
        np.nan
    )

    # =========================================================================
    # FIGURE 1: 2D TOPOGRAPHIC CONTOUR & PLUME MAP
    # =========================================================================
    fig2d, ax = plt.subplots(figsize=(12, 7))
    im = ax.pcolormesh(LON, LAT, C_GRID, cmap="YlOrRd", shading="auto", alpha=0.68, vmin=bg_val, vmax=vmax_val, zorder=2)

    cs = ax.contour(LON, LAT, ELEV, levels=[10, 25, 50, 100, 200, 300, 400], colors="#2b2b2b", linewidths=0.8, zorder=3)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%d m")

    if gdf_rivers is not None:
        gdf_rivers.plot(ax=ax, color="#0055cc", linewidth=1.6, alpha=0.85, zorder=4, label="OSM Waterway Network")
    else:
        ax.plot(r_lon, r_lat, color="#0055cc", linewidth=2.5, zorder=4, label="Santa Cruz / Bayto River")

    if HAS_GIS:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenTopoMap, alpha=0.70, zorder=1)
        except Exception:
            ax.set_facecolor("#cce5ff")
    else:
        ax.set_facecolor("#cce5ff")

    ax.scatter(
        stations_df["longitude"], stations_df["latitude"],
        c=stations_df["predicted_sed_mg_kg"], cmap="YlOrRd", vmin=bg_val, vmax=vmax_val,
        edgecolor="black", s=90, linewidth=1.3, zorder=6, label="Monitoring Stations"
    )
    for _, row in stations_df.iterrows():
        ax.annotate(
            f"{row['location']}\n({row['predicted_sed_mg_kg']:.1f} mg/kg)",
            (row["longitude"], row["latitude"]),
            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.88), zorder=7
        )

    cbar2d = fig2d.colorbar(im, ax=ax, pad=0.02)
    cbar2d.set_label(f"Accumulated {analyte_name} in Sediment (mg/kg) after {time_value} {time_unit}", fontweight="bold")
    ax.set_title(
        f"Santa Cruz River Basin — 2D Predictive Topographic & Plume Map ({time_value} {time_unit})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower left")

    out_2d = OUTPUT_DIR / f"predictive_2d_map_{time_value}_{time_unit}.png"
    plt.tight_layout()
    fig2d.savefig(str(out_2d), dpi=300)
    print(f"Saved 2D predictive map to: {out_2d.resolve()}")

    # =========================================================================
    # FIGURE 2: TRUE 3D TERRAIN SURFACE + PSEUDO-3D ELEVATION-TIERED LAYERS
    # =========================================================================
    if show_3d:
        C_GRID_3D = np.nan_to_num(C_GRID, nan=bg_val)
        norm_c = Normalize(vmin=bg_val, vmax=vmax_val)
        terrain_norm = Normalize(vmin=0.0, vmax=max(float(ELEV.max()), 1.0))

        fig3d = plt.figure(figsize=(16, 7.5))

        # --- Panel A: True 3D Topographic Surface + 3D River & Draped Plume ---
        ax3d = fig3d.add_subplot(1, 2, 1, projection="3d")
        terrain_rgba = cm.terrain(terrain_norm(ELEV))
        plume_rgba = cm.YlOrRd(norm_c(C_GRID_3D))

        rel_intensity = np.clip((C_GRID_3D - bg_val) / max(vmax_val - bg_val, 1e-4), 0.0, 1.0)
        alpha_plume = (rel_intensity ** 0.65 * 0.92)[..., None]
        facecolors = (1.0 - alpha_plume) * terrain_rgba + alpha_plume * plume_rgba
        facecolors[ELEV <= 0.5] = np.array([0.75, 0.88, 1.0, 1.0])

        ax3d.plot_surface(
            LON, LAT, ELEV,
            facecolors=facecolors, rstride=2, cstride=2,
            antialiased=True, shade=True
        )
        ax3d.plot(r_lon, r_lat, r_elev + 10.0, color="#0044cc", linewidth=3.0, label="3D Santa Cruz River", zorder=10)

        for _, row in stations_df.iterrows():
            idx = int(row["node_idx"])
            ax3d.scatter(r_lon[idx], r_lat[idx], r_elev[idx] + 16.0, color="cyan", edgecolor="black", s=50, zorder=15)
            ax3d.text(
                r_lon[idx], r_lat[idx], r_elev[idx] + 30.0,
                f"{row['location'].split('(')[0].strip()}\n({row['predicted_sed_mg_kg']:.0f})",
                fontsize=6.8, fontweight="bold", ha="center"
            )

        ax3d.view_init(elev=34, azim=-118)
        ax3d.set_title(
            f"True 3D Terrain & River Plume\n(After {time_value} {time_unit})",
            fontsize=11, fontweight="bold"
        )
        ax3d.set_xlabel("Longitude (°E)", fontsize=8.5, labelpad=6)
        ax3d.set_ylabel("Latitude (°N)", fontsize=8.5, labelpad=6)
        ax3d.set_zlabel("Elevation (m ASL)", fontsize=8.5, labelpad=4)

        # --- Panel B: Pseudo-3D Stacked Elevation-Interval Layers ---
        ax_p3d = fig3d.add_subplot(1, 2, 2, projection="3d")
        elev_intervals = [(0, 25), (25, 75), (75, 150), (150, 250), (250, 500)]

        for low_e, high_e in elev_intervals:
            layer_mask = (ELEV >= low_e) & (ELEV < high_e)
            if not np.any(layer_mask):
                continue
            Z_slice = np.full_like(ELEV, float(low_e))
            C_slice = np.where(layer_mask, C_GRID_3D, bg_val)
            fc_slice = cm.YlOrRd(norm_c(C_slice))
            fc_slice[..., 3] = np.where(layer_mask, 0.85, 0.0)

            ax_p3d.plot_surface(
                LON, LAT, Z_slice,
                facecolors=fc_slice, rstride=2, cstride=2, shade=False
            )
            if low_e > 0 and ELEV.min() < low_e < ELEV.max():
                ax_p3d.contour(LON, LAT, ELEV, levels=[low_e], zdir="z", offset=float(low_e), colors="#2b2b2b", linewidths=0.9)

        ax_p3d.plot(r_lon, r_lat, r_elev, color="#0055cc", linewidth=2.8, label="3D River Profile")
        ax_p3d.view_init(elev=32, azim=-115)
        ax_p3d.set_title(
            "Pseudo-3D Stacked Elevation-Interval Layers\n(0–25m, 25–75m, 75–150m, 150–250m, >250m)",
            fontsize=11, fontweight="bold"
        )
        ax_p3d.set_xlabel("Longitude (°E)", fontsize=8.5, labelpad=6)
        ax3d.set_ylabel("Latitude (°N)", fontsize=8.5, labelpad=6)
        ax_p3d.set_zlabel("Elevation Tier (m)", fontsize=8.5, labelpad=4)

        sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm_c)
        sm.set_array([])
        cbar3d = fig3d.colorbar(sm, ax=[ax3d, ax_p3d], shrink=0.68, pad=0.04)
        cbar3d.set_label(f"Accumulated {analyte_name} in Sediment (mg/kg)", fontweight="bold")

        out_3d = OUTPUT_DIR / f"predictive_3d_and_pseudo3d_{time_value}_{time_unit}.png"
        fig3d.savefig(str(out_3d), dpi=300, bbox_inches="tight")
        print(f"Saved 3D & Pseudo-3D map to: {out_3d.resolve()}")

    plt.show()


if __name__ == "__main__":
    # Flexible time arguments: e.g., time_value=14, time_unit="days" (or "hours", "minutes")
    generate_predictive_map(time_value=5, time_unit="minutes")
    generate_predictive_map(time_value=15, time_unit="minutes")
    generate_predictive_map(time_value=30, time_unit="minutes")
    generate_predictive_map(time_value=45, time_unit="minutes")
    generate_predictive_map(time_value=60, time_unit="minutes")
    generate_predictive_map(time_value=30, time_unit="days")
    generate_predictive_map(time_value=60, time_unit="days")

    