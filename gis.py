"""GIS utilities for the Zambales Ni-Cr analysis project.

Commands
--------
Plot sample data over a georeferenced DEM:
    python gis.py plot --dem data/dem.tif --samples data/prepared_data.csv \
        --analyte Ni --sample-type water --unit mg/L --output gis_map.png

Convert a grayscale PNG heightmap to a georeferenced GeoTIFF:
    python gis.py convert-heightmap terrain.png data/dem.tif \
        --bounds WEST SOUTH EAST NORTH --crs EPSG:4326 \
        --elevation-min 0 --elevation-max 1800

Heightmap conversion is only valid if the PNG's grayscale values linearly encode
height. The geographic bounds, CRS, and elevation range must come from reliable
source metadata; this script cannot infer them from the image alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import CRS as PyprojCRS
from rasterio.transform import from_bounds


def heightmap_to_geotiff(
    png_path: str | Path,
    output_path: str | Path,
    bounds: tuple[float, float, float, float],
    crs: str,
    elevation_min: float,
    elevation_max: float,
    white_is_low: bool = False,
) -> Path:
    """Convert a linear grayscale elevation PNG into a single-band GeoTIFF.

    `bounds` is (west, south, east, north) in `crs`. By default, black maps to
    elevation_min and white maps to elevation_max. Use white_is_low=True if
    the source uses the inverse encoding.
    """
    png_path, output_path = Path(png_path), Path(output_path)
    if not png_path.is_file():
        raise FileNotFoundError(f"Heightmap not found: {png_path}")
    if len(bounds) != 4:
        raise ValueError("bounds must be (west, south, east, north)")

    west, south, east, north = map(float, bounds)
    if not west < east or not south < north:
        raise ValueError("Invalid bounds: require west < east and south < north")
    if not elevation_min < elevation_max:
        raise ValueError("Require elevation_min < elevation_max")

    with Image.open(png_path) as image:
        # Do not silently convert RGB artwork to grayscale: its colors may not
        # encode elevation at all.
        if image.mode == "L":
            max_code = 255
        elif image.mode in ("I;16", "I;16L", "I;16B"):
            max_code = 65535
        else:
            raise ValueError(
                f"Unsupported PNG mode {image.mode!r}; expected 8-bit ('L') or "
                "16-bit grayscale. RGB/color images are not treated as elevation."
            )
        pixels = np.asarray(image).astype(np.float32)

    fraction = pixels / max_code
    if white_is_low:
        fraction = 1.0 - fraction
    elevation = elevation_min + fraction * (elevation_max - elevation_min)
    transform = from_bounds(west, south, east, north, elevation.shape[1], elevation.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(elevation, 1)
        dst.set_band_description(1, "Elevation")
        dst.update_tags(
            source_file=png_path.name,
            grayscale_encoding="linear",
            elevation_min=str(elevation_min),
            elevation_max=str(elevation_max),
            white_is_low=str(white_is_low).lower(),
        )

    return output_path


def _normalize_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common headings without discarding unknown columns."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "long": "longitude",
        "sample_no": "sample_id",
        "sampling_date": "date",
        "sampling_season": "season",
        "concentration": "value_numeric",
        "measurement": "value_numeric",
        "result": "value_numeric",
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalizing CSV headings.")
    return df


def _read_vector(path: str | Path, target_crs) -> gpd.GeoDataFrame:
    layer = gpd.read_file(path)
    if layer.crs is None:
        raise ValueError(f"Vector layer has no CRS: {path}")
    return layer.to_crs(target_crs)


def plot_samples(
    dem_path: str | Path,
    samples_path: str | Path,
    analyte: str = "Ni",
    sample_type: str = "water",
    unit: str = "mg/L",
    rivers_path: str | Path | None = None,
    mines_path: str | Path | None = None,
    sample_crs: str = "EPSG:4326",
    output_path: str | Path = "gis_map.png",
    show: bool = True,
):
    """Plot one analyte/sample type/unit over a DEM and optional vector layers.

    The CSV needs latitude/longitude (or lat/lon) and value_numeric. If it also
    contains below_detection_limit, those measurements are mapped as locations
    but excluded from the numeric color scale.
    """
    dem_path, samples_path = Path(dem_path), Path(samples_path)
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not samples_path.is_file():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    df = _normalize_sample_columns(pd.read_csv(samples_path))
    required = {"latitude", "longitude", "analyte", "sample_type", "unit", "value_numeric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Samples CSV is missing required columns: {sorted(missing)}")

    # Keep the comparison on one analyte, sample matrix, and unit.
    for column in ("analyte", "sample_type", "unit"):
        df[column] = df[column].astype("string").str.strip()
    selected = df[
        df["analyte"].str.casefold().eq(analyte.strip().casefold())
        & df["sample_type"].str.casefold().eq(sample_type.strip().casefold())
        & df["unit"].str.casefold().eq(unit.strip().casefold())
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No rows match analyte={analyte!r}, sample_type={sample_type!r}, unit={unit!r}. "
            "Check the CSV values or pass the appropriate command-line options."
        )

    for column in ("latitude", "longitude", "value_numeric"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=["latitude", "longitude"])
    if selected.empty:
        raise ValueError("Matching rows have no valid latitude/longitude coordinates.")
    if not selected["latitude"].between(-90, 90).all():
        raise ValueError("Latitude values must be between -90 and 90 degrees.")
    if not selected["longitude"].between(-180, 180).all():
        raise ValueError("Longitude values must be between -180 and 180 degrees.")

    points = gpd.GeoDataFrame(
        selected,
        geometry=gpd.points_from_xy(selected["longitude"], selected["latitude"]),
        crs=sample_crs,
    )

    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        elevation = dem.read(1, masked=True)
        bounds, dem_crs = dem.bounds, dem.crs

    points = points.to_crs(dem_crs)
    fig, ax = plt.subplots(figsize=(10, 8))
    terrain = ax.imshow(
        elevation,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap="terrain",
    )
    fig.colorbar(terrain, ax=ax, label="Elevation (DEM units)", shrink=0.78)

    if rivers_path:
        _read_vector(rivers_path, dem_crs).plot(
            ax=ax, color="#168bd2", linewidth=1.0, label="Rivers", zorder=2
        )
    if mines_path:
        _read_vector(mines_path, dem_crs).plot(
            ax=ax, facecolor="none", edgecolor="#d62728", linewidth=1.2,
            label="Mining areas", zorder=2
        )

    # Draw every matching point first, including values below detection limits.
    ax.scatter(
        points.geometry.x,
        points.geometry.y,
        s=36,
        facecolors="white",
        edgecolors="black",
        linewidths=0.7,
        label="Sample locations",
        zorder=3,
    )

    detected = points["value_numeric"].notna()
    if "below_detection_limit" in points.columns:
        flagged = points["below_detection_limit"].astype("string").str.casefold()
        detected &= ~flagged.isin(["true", "1", "yes", "y"])

    if detected.any():
        measured = points.loc[detected]
        scatter = ax.scatter(
            measured.geometry.x,
            measured.geometry.y,
            c=measured["value_numeric"],
            cmap="plasma",
            s=52,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax, label=f"{analyte} ({unit})", shrink=0.78)

    ax.set_title(f"{analyte} in {sample_type} samples")
    axes = PyprojCRS.from_user_input(dem_crs).axis_info
    x_unit = axes[0].unit_name if axes else "map units"
    y_unit = axes[1].unit_name if len(axes) > 1 else x_unit
    ax.set_xlabel(f"Map X ({x_unit})")
    ax.set_ylabel(f"Map Y ({y_unit})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GIS helpers for terrain and contaminant samples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plot sample measurements over a DEM.")
    plot_parser.add_argument("--dem", required=True, help="Georeferenced DEM GeoTIFF path.")
    plot_parser.add_argument("--samples", required=True, help="Prepared sample CSV path.")
    plot_parser.add_argument("--analyte", default="Ni")
    plot_parser.add_argument("--sample-type", default="water")
    plot_parser.add_argument("--unit", default="mg/L")
    plot_parser.add_argument("--rivers", help="Optional river vector layer.")
    plot_parser.add_argument("--mines", help="Optional mining-area vector layer.")
    plot_parser.add_argument("--sample-crs", default="EPSG:4326", help="CRS of sample coordinates.")
    plot_parser.add_argument("--output", default="gis_map.png")
    plot_parser.add_argument("--no-show", action="store_true", help="Save the map without opening a window.")

    convert_parser = subparsers.add_parser(
        "convert-heightmap", help="Convert linearly encoded grayscale PNG to GeoTIFF."
    )
    convert_parser.add_argument("png", help="Grayscale heightmap PNG.")
    convert_parser.add_argument("output", help="Output GeoTIFF path.")
    convert_parser.add_argument(
        "--bounds", nargs=4, type=float, required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Geographic/spatial bounds in the selected CRS.",
    )
    convert_parser.add_argument("--crs", required=True, help="CRS for the supplied bounds.")
    convert_parser.add_argument("--elevation-min", type=float, required=True, help="Elevation represented by black.")
    convert_parser.add_argument("--elevation-max", type=float, required=True, help="Elevation represented by white.")
    convert_parser.add_argument(
        "--white-is-low", action="store_true",
        help="Reverse the height mapping if white represents the minimum elevation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plot":
        plot_samples(
            dem_path=args.dem,
            samples_path=args.samples,
            analyte=args.analyte,
            sample_type=args.sample_type,
            unit=args.unit,
            rivers_path=args.rivers,
            mines_path=args.mines,
            sample_crs=args.sample_crs,
            output_path=args.output,
            show=not args.no_show,
        )
        print(f"Saved map to {args.output}")
    elif args.command == "convert-heightmap":
        output = heightmap_to_geotiff(
            png_path=args.png,
            output_path=args.output,
            bounds=tuple(args.bounds),
            crs=args.crs,
            elevation_min=args.elevation_min,
            elevation_max=args.elevation_max,
            white_is_low=args.white_is_low,
        )
        print(f"Saved GeoTIFF to {output}")


if __name__ == "__main__":
    main()
