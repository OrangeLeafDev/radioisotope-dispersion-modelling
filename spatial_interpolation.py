import os
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Ensure the output directory exists on Windows/macOS/Linux
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional GIS & Basemap libraries
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point, LineString
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


def fetch_real_river_and_snap_stations(stations_df):
    """
    Downloads the Bayto / Cabaluan / Santa Cruz River network from OpenStreetMap
    and snaps stations to the active channel. Gracefully falls back if offline.
    """
    gdf_rivers = None
    if HAS_GIS:
        try:
            bbox = (119.885, 15.685, 120.040, 15.745)  # (west, south, east, north)
            tags = {"waterway": ["river", "stream"]}
            print("Downloading river network from OpenStreetMap for Santa Cruz, Zambales...")
            gdf_rivers = ox.features_from_bbox(bbox=bbox, tags=tags)
            gdf_rivers = gdf_rivers[gdf_rivers.geometry.type.isin(["LineString", "MultiLineString"])]

            # Use union_all() for modern Shapely 2.0+, unary_union for older versions
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


def get_real_elevation_grid(LON, LAT, dem_tif_path="srtm_santa_cruz.tif"):
    """
    1. Loads local SRTM DEM raster (.tif) if available.
    2. Queries Open-Meteo Elevation API if connected to the internet.
    3. Falls back to an analytical Zambales Ophiolite elevation surface if offline.
    """
    # 1. Local DEM TIF file
    if HAS_RASTERIO and os.path.exists(dem_tif_path):
        try:
            print(f"Loading local DEM contours from {dem_tif_path}...")
            with rasterio.open(dem_tif_path) as src:
                coords = [(lon, lat) for lon, lat in zip(LON.ravel(), LAT.ravel())]
                elev = np.array([val[0] for val in src.sample(coords)]).reshape(LON.shape)
                return np.where(elev < -100, 0, elev)
        except Exception as e:
            print(f"Failed to read local DEM file: {e}. Falling back to API/model.")

    # 2. Try Open-Meteo DEM API
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
        return rbf_dem(np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)
    except Exception as e:
        print(f"Notice: Elevation API unavailable ({e}). Generating analytical Zambales topography surface.")

    # 3. Offline Analytical Topographic Model (Coast to Zambales Ophiolite)
    coast_lon = 119.892
    land_dist = np.maximum(0.0, LON - coast_lon)
    base_elev = np.where(LON > coast_lon, 450.0 * (land_dist / 0.14) ** 1.4, -5.0)
    valley_axis = 15.714 + 0.05 * (LON - coast_lon)
    valley_carve = 1.0 - 0.45 * np.exp(-((LAT - valley_axis) / 0.015) ** 2)
    return np.maximum(0.5, base_elev * valley_carve)


def plot_real_map_kriging(dem_tif_path="srtm_santa_cruz.tif"):
    # Station coordinates along the Bayto / Santa Cruz River
    stations_df = pd.DataFrame([
        {"location": "Station 00 (Control)",   "longitude": 120.025000, "latitude": 15.699722, "Ni": 0.018},
        {"location": "Source (Laterite Mine)", "longitude": 120.000000, "latitude": 15.699444, "Ni": 0.850},
        {"location": "Near-Field",             "longitude": 119.989000, "latitude": 15.704200, "Ni": 0.540},
        {"location": "Mid-Field",              "longitude": 119.964500, "latitude": 15.713500, "Ni": 0.260},
        {"location": "Lower Reach (15.719°N)", "longitude": 119.937874, "latitude": 15.719390, "Ni": 0.175},
        {"location": "Outlet (River Mouth)",   "longitude": 119.892000, "latitude": 15.714300, "Ni": 0.125},
    ])

    # 1. Snap stations to river channel geometry
    stations_df, gdf_rivers = fetch_real_river_and_snap_stations(stations_df)

    # 2. Grid bounds for Santa Cruz River Basin
    grid_lon = np.linspace(119.885, 120.035, 200)
    grid_lat = np.linspace(15.685, 15.740, 150)
    LON, LAT = np.meshgrid(grid_lon, grid_lat)

    # 3. Topographic elevation surface
    ELEV = get_real_elevation_grid(LON, LAT, dem_tif_path=dem_tif_path)

    # 4. Spatial Interpolation along the river corridor
    pts = stations_df[["longitude", "latitude"]].values
    vals = stations_df["Ni"].values

    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-3)
    z_interp = rbf(np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)

    t_wp = np.linspace(0, 1, len(pts))
    t_fine = np.linspace(0, 1, 250)
    r_lon = np.interp(t_fine, t_wp, pts[::-1, 0])
    r_lat = np.interp(t_fine, t_wp, pts[::-1, 1])
    dist_to_river = np.min(
        np.hypot(LON[:, :, None] - r_lon[None, None, :], LAT[:, :, None] - r_lat[None, None, :]),
        axis=2
    )
    corridor_weight = np.exp(-(dist_to_river / 0.012) ** 1.5)
    C_GRID = np.where(ELEV > 0.5, np.clip(z_interp * corridor_weight, 0, vals.max()), np.nan)

    # 5. Render map
    fig, ax = plt.subplots(figsize=(12, 7))

    # Dispersion plume heatmap
    im = ax.pcolormesh(LON, LAT, C_GRID, cmap="YlOrRd", shading="auto", alpha=0.62, vmin=0, vmax=0.85, zorder=2)

    # Elevation contours
    cs = ax.contour(LON, LAT, ELEV, levels=[10, 25, 50, 100, 200, 300, 400], colors="#2b2b2b", linewidths=0.8, zorder=3)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%d m")

    # Waterways
    if gdf_rivers is not None:
        gdf_rivers.plot(ax=ax, color="#0055cc", linewidth=1.6, alpha=0.85, zorder=4, label="OSM Waterway Network")
    else:
        ax.plot(r_lon, r_lat, color="#0055cc", linewidth=2.5, zorder=4, label="Santa Cruz / Bayto River")

    # Basemap tiles (if connected and contextily is installed)
    if HAS_GIS:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenTopoMap, alpha=0.70, zorder=1)
        except Exception:
            ax.set_facecolor("#e6f2ff")
    else:
        ax.set_facecolor("#e6f2ff")

    # Monitoring stations
    ax.scatter(
        stations_df["longitude"], stations_df["latitude"],
        c=stations_df["Ni"], cmap="YlOrRd", vmin=0, vmax=0.85,
        edgecolor="black", s=90, linewidth=1.3, zorder=6, label="Monitoring Stations"
    )
    for _, row in stations_df.iterrows():
        ax.annotate(
            f"{row['location']}\n({row['Ni']:.3f} mg/L)",
            (row["longitude"], row["latitude"]),
            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.88), zorder=7
        )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Nickel (Ni) Concentration (mg/L)", fontweight="bold")
    ax.set_title("Santa Cruz River Basin — Elevation Contours & Kriging Plume Map", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower left")

    out_file = OUTPUT_DIR / "real_map_santa_cruz_kriging.png"
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    print(f"Map successfully rendered and saved to {out_file.resolve()}")
    plt.show()


if __name__ == "__main__":
    plot_real_map_kriging()