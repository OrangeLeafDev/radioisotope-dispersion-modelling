import os
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator

from fileio import read_table
from dataproc import prepare_data

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_stations_from_dataset(analyte_name="Nickel (Ni)", storm_phase="During storm"):
    """
    Loads empirical/synthetic station values from santa_cruz_enriched_dataset.csv or Excel if present.
    """
    df_raw = None
    if Path("santa_cruz_enriched_dataset.csv").exists():
        df_raw = read_table("santa_cruz_enriched_dataset.csv")
    else:
        for excel_name in ("SantaCruz_NiCr_Synthetic_Dataset_2.xlsx", "SantaCruz_NiCr_Synthetic_Dataset.xlsx"):
            if Path(excel_name).exists():
                df_raw = read_table(excel_name, sheet_name="Data")
                break

    stations_df = pd.DataFrame([
        {"site_id": "S0", "location": "Station 00 (Control)",   "longitude": 120.025000, "latitude": 15.699722, "node_idx": 0,   "water_mg_L": 0.006,  "sed_mg_kg": 18.0},
        {"site_id": "S1", "location": "Source (Laterite Mine)", "longitude": 120.000000, "latitude": 15.699444, "node_idx": 25,  "water_mg_L": 0.1818, "sed_mg_kg": 22.0},
        {"site_id": "NF", "location": "Near-Field",             "longitude": 119.989000, "latitude": 15.704200, "node_idx": 60,  "water_mg_L": 0.1250, "sed_mg_kg": 20.0},
        {"site_id": "S2", "location": "Mid-Field",              "longitude": 119.964500, "latitude": 15.713500, "node_idx": 110, "water_mg_L": 0.0716, "sed_mg_kg": 19.0},
        {"site_id": "LR", "location": "Lower Reach (15.719°N)", "longitude": 119.937874, "latitude": 15.719390, "node_idx": 150, "water_mg_L": 0.0450, "sed_mg_kg": 18.5},
        {"site_id": "S3", "location": "Outlet (River Mouth)",   "longitude": 119.892000, "latitude": 15.714300, "node_idx": 199, "water_mg_L": 0.0243, "sed_mg_kg": 18.0},
    ])

    if df_raw is not None:
        clean = prepare_data(df_raw)
        sub = clean[
            (clean["analyte"].str.lower() == analyte_name.lower()) &
            (clean["storm_phase"].str.lower() == storm_phase.lower())
        ]
        for idx, row in stations_df.iterrows():
            sid = row["site_id"].lower()
            w_vals = sub[(sub["location"].str.lower() == sid) & (sub["sample_type"].str.lower() == "water")]["value_numeric"]
            if len(w_vals) > 0:
                stations_df.loc[idx, "water_mg_L"] = float(w_vals.mean())

    return stations_df


def fetch_real_river_and_snap_stations(stations_df):
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
    if HAS_RASTERIO and os.path.exists(dem_tif_path):
        try:
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
        flat_lats, flat_lons = SLAT.ravel(), SLON.ravel()
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


def compute_channel_hydraulics(H_depth, W_width=25.0, manning_n=0.040, S0_slope=0.0025):
    """Computes Manning velocity v (m/s) and Fischer dispersion D (m^2/s) for a given water depth H."""
    R_hyd = (W_width * H_depth) / (W_width + 2.0 * H_depth)
    v_flow = (1.0 / manning_n) * (R_hyd ** (2.0 / 3.0)) * (S0_slope ** 0.5)
    u_star = np.sqrt(9.81 * H_depth * S0_slope)
    D_disp = 0.011 * (v_flow ** 2) * (W_width ** 2) / (H_depth * u_star)
    return v_flow, D_disp


def simulate_multi_snapshots_with_events(
    snapshots, events=None, nx=200, dx=50.0,
    manning_n=0.040, W_width=25.0, H_base=1.3, S0_slope=0.0025,
    Kd_metal=1500.0, ssc_base=85.0, v_settle=1.5e-4, bg_sediment_mg_kg=18.0
):
    """
    Runs a SINGLE continuous simulation pass up to max(snapshots), capturing snapshots
    along the way while dynamically applying scheduled Habagat/monsoon or tributary events.
    """
    if events is None:
        events = []

    # 1. Convert all snapshot checkpoints to seconds & sort chronologically
    snap_list = []
    for val, unit in snapshots:
        sec = get_time_in_seconds(val, unit)
        snap_list.append({"val": val, "unit": unit, "seconds": sec})
    snap_list.sort(key=lambda x: x["seconds"])
    max_seconds = snap_list[-1]["seconds"]

    # 2. Pre-process scheduled dispersion/monsoon events into seconds
    parsed_events = []
    max_H = H_base
    for ev in events:
        start_s = get_time_in_seconds(*ev["start"])
        dur_s = get_time_in_seconds(*ev["duration"])
        end_s = start_s + dur_s
        H_ev = float(ev.get("H_depth", H_base * 1.5))
        max_H = max(max_H, H_ev)
        parsed_events.append({
            "name": ev.get("name", "Scheduled Event"),
            "start_s": start_s,
            "end_s": end_s,
            "node_idx": int(ev.get("node_idx", 25)),
            "H_depth": H_ev,
            "ssc_mg_L": float(ev.get("ssc_mg_L", 550.0)),
            "source_load": float(ev.get("source_load", 0.030)),
        })

    # 3. Guarantee Courant & Diffusion stability across the highest storm flow velocity
    v_max, D_max = compute_channel_hydraulics(max_H, W_width, manning_n, S0_slope)
    dt_max_adv = 0.45 * dx / max(v_max, 1e-6)
    dt_max_diff = 0.20 * (dx ** 2) / max(D_max, 1e-6)
    dt = min(15.0, dt_max_adv, dt_max_diff)

    nt = max(1, int(np.ceil(max_seconds / dt)))
    print(f"Running single continuous pass: {nt:,} steps (dt = {dt:.2f} s) up to {snap_list[-1]['val']} {snap_list[-1]['unit']}...")

    # Pre-calculate baseline hydraulics & partitioning
    v_base, D_base = compute_channel_hydraulics(H_base, W_width, manning_n, S0_slope)
    f_d_base = 1.0 / (1.0 + Kd_metal * (ssc_base * 1e-6))
    f_p_base = 1.0 - f_d_base
    k_settle_base = f_p_base * (v_settle / H_base)

    C_water = np.full(nx, 0.012)
    Bed_Sediment = np.full(nx, bg_sediment_mg_kg)
    bed_conversion = 0.085

    captured_results = {}
    snap_ptr = 0

    # Capture any t = 0 snapshots immediately
    while snap_ptr < len(snap_list) and snap_list[snap_ptr]["seconds"] <= 0.0:
        s_info = snap_list[snap_ptr]
        captured_results[(s_info["val"], s_info["unit"])] = Bed_Sediment.copy()
        print(f"  -> Captured snapshot: {s_info['val']} {s_info['unit']}")
        snap_ptr += 1

    # 4. Single-pass time integration loop
    for t in range(1, nt + 1):
        curr_time_s = t * dt

        # Check active events at current time
        H_curr = H_base
        ssc_curr = ssc_base
        extra_loads = []

        for pev in parsed_events:
            if pev["start_s"] <= curr_time_s <= pev["end_s"]:
                H_curr = max(H_curr, pev["H_depth"])
                ssc_curr = max(ssc_curr, pev["ssc_mg_L"])
                extra_loads.append((pev["node_idx"], pev["source_load"]))

        if H_curr != H_base or ssc_curr != ssc_base:
            v_curr, D_curr = compute_channel_hydraulics(H_curr, W_width, manning_n, S0_slope)
            f_d_curr = 1.0 / (1.0 + Kd_metal * (ssc_curr * 1e-6))
            f_p_curr = 1.0 - f_d_curr
            k_settle_curr = f_p_curr * (v_settle / H_curr)
        else:
            v_curr, D_curr, k_settle_curr = v_base, D_base, k_settle_base

        prev_C = C_water.copy()
        d2Cdx2 = (prev_C[2:] - 2.0 * prev_C[1:-1] + prev_C[:-2]) / (dx ** 2)
        dCdx = (prev_C[1:-1] - prev_C[:-2]) / dx

        C_water[1:-1] = (
            prev_C[1:-1]
            + dt * (D_curr * d2Cdx2 - v_curr * dCdx - k_settle_curr * prev_C[1:-1])
        )

        # Continuous baseline mine seepage at primary source node (node 25)
        C_water[25] += 0.0004 * dt

        # Inject any active Habagat / secondary dispersion sources
        for node_i, load_val in extra_loads:
            C_water[node_i] += load_val * dt

        C_water[0] = 0.012
        C_water[-1] = C_water[-2]
        Bed_Sediment += (k_settle_curr * C_water * dt * Kd_metal) * bed_conversion

        # Check if we crossed any requested snapshot timestamps
        while snap_ptr < len(snap_list) and curr_time_s >= snap_list[snap_ptr]["seconds"]:
            s_info = snap_list[snap_ptr]
            captured_results[(s_info["val"], s_info["unit"])] = Bed_Sediment.copy()
            print(f"  -> Captured snapshot: {s_info['val']} {s_info['unit']} (Max Bed Ni = {Bed_Sediment.max():.1f} mg/kg)")
            snap_ptr += 1

    return captured_results


def render_and_save_snapshot(
    final_bed, time_value, time_unit, stations_df, gdf_rivers,
    LON, LAT, ELEV, r_lon, r_lat, r_elev, dist_to_river, corridor_weight,
    analyte_name="Nickel (Ni)", show_3d=True, display_windows=False
):
    """Renders and exports the 2D and 3D/Pseudo-3D maps for a captured snapshot."""
    stations_df = stations_df.copy()
    stations_df["predicted_sed_mg_kg"] = final_bed[stations_df["node_idx"].values]

    rbf_plume = RBFInterpolator(np.column_stack([r_lon, r_lat]), final_bed, kernel="thin_plate_spline", smoothing=0.5)
    z_interp = rbf_plume(np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)

    bg_val = float(final_bed.min())
    vmax_val = max(float(final_bed.max()), bg_val + 1.0)
    C_GRID = np.where(
        ELEV > 0.5,
        np.clip(z_interp * corridor_weight + bg_val * (1.0 - corridor_weight), bg_val, vmax_val),
        np.nan
    )

    # 1. 2D Map
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
    if not display_windows:
        plt.close(fig2d)

    # 2. 3D & Pseudo-3D Map
    if show_3d:
        C_GRID_3D = np.nan_to_num(C_GRID, nan=bg_val)
        norm_c = Normalize(vmin=bg_val, vmax=vmax_val)
        terrain_norm = Normalize(vmin=0.0, vmax=max(float(ELEV.max()), 1.0))

        fig3d = plt.figure(figsize=(16, 7.5))

        ax3d = fig3d.add_subplot(1, 2, 1, projection="3d")
        terrain_rgba = cm.terrain(terrain_norm(ELEV))
        plume_rgba = cm.YlOrRd(norm_c(C_GRID_3D))

        rel_intensity = np.clip((C_GRID_3D - bg_val) / max(vmax_val - bg_val, 1e-4), 0.0, 1.0)
        alpha_plume = (rel_intensity ** 0.65 * 0.92)[..., None]
        facecolors = (1.0 - alpha_plume) * terrain_rgba + alpha_plume * plume_rgba
        facecolors[ELEV <= 0.5] = np.array([0.75, 0.88, 1.0, 1.0])

        ax3d.plot_surface(LON, LAT, ELEV, facecolors=facecolors, rstride=2, cstride=2, antialiased=True, shade=True)
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
        ax3d.set_title(f"True 3D Terrain & River Plume\n(After {time_value} {time_unit})", fontsize=11, fontweight="bold")
        ax3d.set_xlabel("Longitude (°E)", fontsize=8.5, labelpad=6)
        ax3d.set_ylabel("Latitude (°N)", fontsize=8.5, labelpad=6)
        ax3d.set_zlabel("Elevation (m ASL)", fontsize=8.5, labelpad=4)

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

            ax_p3d.plot_surface(LON, LAT, Z_slice, facecolors=fc_slice, rstride=2, cstride=2, shade=False)
            if low_e > 0 and ELEV.min() < low_e < ELEV.max():
                ax_p3d.contour(LON, LAT, ELEV, levels=[low_e], zdir="z", offset=float(low_e), colors="#2b2b2b", linewidths=0.9)

        ax_p3d.plot(r_lon, r_lat, r_elev, color="#0055cc", linewidth=2.8, label="3D River Profile")
        ax_p3d.view_init(elev=32, azim=-115)
        ax_p3d.set_title(
            "Pseudo-3D Stacked Elevation-Interval Layers\n(0–25m, 25–75m, 75–150m, 150–250m, >250m)",
            fontsize=11, fontweight="bold"
        )
        ax_p3d.set_xlabel("Longitude (°E)", fontsize=8.5, labelpad=6)
        ax_p3d.set_ylabel("Latitude (°N)", fontsize=8.5, labelpad=6)
        ax_p3d.set_zlabel("Elevation Tier (m)", fontsize=8.5, labelpad=4)

        sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm_c)
        sm.set_array([])
        cbar3d = fig3d.colorbar(sm, ax=[ax3d, ax_p3d], shrink=0.68, pad=0.04)
        cbar3d.set_label(f"Accumulated {analyte_name} in Sediment (mg/kg)", fontweight="bold")

        out_3d = OUTPUT_DIR / f"predictive_3d_and_pseudo3d_{time_value}_{time_unit}.png"
        fig3d.savefig(str(out_3d), dpi=300, bbox_inches="tight")
        if not display_windows:
            plt.close(fig3d)

    if display_windows:
        plt.show()
    print()


def generate_predictive_snapshots(
    snapshots, events=None, analyte_name="Nickel (Ni)",
    dem_tif_path="srtm_santa_cruz.tif", show_3d=True, display_windows=False
):
    """
    Master pipeline:
    1. Builds the river & DEM terrain grid ONCE.
    2. Runs a SINGLE continuous simulation pass with scheduled Habagat/dispersion events.
    3. Automatically exports 2D and 3D maps for every snapshot in `snapshots`.
    """
    stations_df = load_stations_from_dataset(analyte_name=analyte_name)
    stations_df, gdf_rivers = fetch_real_river_and_snap_stations(stations_df)

    nx = 200
    pts_wp = stations_df[["longitude", "latitude"]].values
    t_wp = np.linspace(0, 1, len(pts_wp))
    t_fine = np.linspace(0, 1, nx)
    r_lon = np.interp(t_fine, t_wp, pts_wp[:, 0])
    r_lat = np.interp(t_fine, t_wp, pts_wp[:, 1])

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

    dist_to_river = np.min(
        np.hypot(LON[:, :, None] - r_lon[None, None, :], LAT[:, :, None] - r_lat[None, None, :]),
        axis=2
    )
    corridor_weight = np.exp(-(dist_to_river / 0.012) ** 1.5)

    # Run all snapshots in ONE pass
    captured = simulate_multi_snapshots_with_events(snapshots=snapshots, events=events, nx=nx, dx=50.0)

    # Render each snapshot
    for (val, unit), bed_profile in captured.items():
        render_and_save_snapshot(
            final_bed=bed_profile, time_value=val, time_unit=unit,
            stations_df=stations_df, gdf_rivers=gdf_rivers,
            LON=LON, LAT=LAT, ELEV=ELEV, r_lon=r_lon, r_lat=r_lat, r_elev=r_elev,
            dist_to_river=dist_to_river, corridor_weight=corridor_weight,
            analyte_name=analyte_name, show_3d=show_3d, display_windows=display_windows
        )
        print(f"! Rendered snapshot: {val} {unit}")
    print(f"All {len(captured)} snapshots exported to {OUTPUT_DIR.resolve()}!")


if __name__ == "__main__":
    # 1. Define all snapshots you want to capture in a SINGLE simulation pass:
    SNAPSHOT_SCHEDULE = [
        (0, "days"),
        (15, "minutes"),
        (60, "minutes"),
        (6, "hours"),
        (1, "days"),
        (5, "days"),
        (7, "days"),
        (15, "days"),
        (20, "days"),
        (30, "days"),
        (40, "days"),
        (50, "days"),
        (60, "days"),
        (70, "days"),
        (80, "days"),
        (90, "days"),  # 3-month mark
    ]

    # 2. Define scheduled Habagat / Monsoon events & secondary dispersion sources:
    #    - 'start': (value, unit) when the event begins
    #    - 'duration': (value, unit) how long the event lasts
    #    - 'node_idx': where the source enters (25 = Mine Source, 85 = Mid-Basin Tributary, etc.)
    #    - 'H_depth': river depth (m) during the event (increases flow velocity & dispersion)
    #    - 'ssc_mg_L': suspended sediment concentration (mg/L) during the event
    #    - 'source_load': metal loading rate (mg/L/s) injected during the event
    DISPERSION_EVENTS = [
        {
            "name": "Initial First-Flush Rainfall",
            "start": (10, "minutes"),
            "duration": (3, "hours"),
            "node_idx": 25,
            "H_depth": 1.9,
            "ssc_mg_L": 450.0,
            "source_load": 0.030,
        },
        {
            "name": "Early Habagat Squall",
            "start": (10, "days"),
            "duration": (18, "hours"),
            "node_idx": 25,
            "H_depth": 2.2,
            "ssc_mg_L": 650.0,
            "source_load": 0.045,
        },
        {
            "name": "Peak Southwest Monsoon (Habagat) Surge",
            "start": (35, "days"),
            "duration": (4, "days"),
            "node_idx": 25,
            "H_depth": 2.7,
            "ssc_mg_L": 950.0,
            "source_load": 0.075,
        },
        {
            "name": "Secondary Tributary Tailings Seepage (Mid-Reach)",
            "start": (50, "days"),
            "duration": (2, "days"),
            "node_idx": 85,   # Enters further downstream!
            "H_depth": 2.0,
            "ssc_mg_L": 500.0,
            "source_load": 0.040,
        },
    ]

    generate_predictive_snapshots(
        snapshots=SNAPSHOT_SCHEDULE,
        events=DISPERSION_EVENTS,
        display_windows=False  # Saves all PNGs quietly without blocking on popup windows
    )