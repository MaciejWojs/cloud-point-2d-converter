#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps, lru_cache
import os
import laspy

# ---------------------
# Dekorator mierzący czas
# ---------------------
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[BENCH] {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ---------------------
# Loader chmur punktów
# ---------------------
@lru_cache(maxsize=8)
@measure_time
def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".las", ".laz"):
        las = laspy.open(path, mode='r')
        points = las.read()
        xyz = np.vstack((points.x, points.y, points.z)).T.astype(np.float32)
        if hasattr(points, "intensity"):
            intensity = np.asarray(points.intensity, dtype=np.float32)
        else:
            intensity = np.ones(xyz.shape[0], dtype=np.float32)
        return np.hstack([xyz, intensity[:, None]])

    elif ext in [".ply", ".pcd", ".xyz", ".obj", ".off"]:
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((len(points),1), dtype=np.float32)])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
            intensity = colors.mean(axis=1, keepdims=True)
            points = np.hstack([points, intensity])
        return points

    elif ext in [".csv", ".txt"]:
        return np.loadtxt(path, delimiter=",", dtype=np.float32)

    else:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")

# ---------------------
# Ekstrakcja płaszczyzny
# ---------------------
def extract_plane_numba(xyz, axis_idx):
    c = xyz[:, axis_idx]
    others = np.delete(xyz, axis_idx, axis=1)
    return others[:,0], others[:,1], c

# ---------------------
# Rasteryzacja chmury do obrazu
# ---------------------
@measure_time
def lidar_to_image(points, axis="z", res=None, target_size=512, limits=None):
    xyz = points[:, :3].astype(np.float32)

    if limits is None:
        mins = xyz.min(0)
        maxs = xyz.max(0)
        limits = {
            "x": (mins[0], maxs[0]),
            "y": (mins[1], maxs[1]),
            "z": (mins[2], maxs[2]),
        }

    if res is None:
        max_range = max(limits["x"][1]-limits["x"][0],
                        limits["y"][1]-limits["y"][0],
                        limits["z"][1]-limits["z"][0])
        res = max_range / target_size

    if axis=="z":
        axis_idx = 2
        lim1, lim2 = limits["x"], limits["y"]
    elif axis=="y":
        axis_idx = 1
        lim1, lim2 = limits["x"], limits["z"]
    else:
        axis_idx = 0
        lim1, lim2 = limits["y"], limits["z"]

    mask = ((xyz[:,0]>=limits["x"][0]) & (xyz[:,0]<=limits["x"][1]) &
            (xyz[:,1]>=limits["y"][0]) & (xyz[:,1]<=limits["y"][1]) &
            (xyz[:,2]>=limits["z"][0]) & (xyz[:,2]<=limits["z"][1]))
    xyz = xyz[mask]

    if len(xyz)==0:
        w = int((lim1[1]-lim1[0])/res)
        h = int((lim2[1]-lim2[0])/res)
        return np.zeros((h,w), dtype=np.uint8), limits, res

    c1, c2, values = extract_plane_numba(xyz, axis_idx)
    xpix = ((c1-lim1[0])/res).astype(np.int32)
    ypix = ((c2-lim2[0])/res).astype(np.int32)

    w = int((lim1[1]-lim1[0])/res)
    h = int((lim2[1]-lim2[0])/res)

    valid = (xpix>=0)&(xpix<w)&(ypix>=0)&(ypix<h)
    xpix = xpix[valid]
    ypix = ypix[valid]
    values = values[valid]

    # ------------------------------
    # Ultraszybka rasteryzacja
    # ------------------------------
    img = np.full((h, w), -np.inf, dtype=np.float32)
    np.maximum.at(img, (ypix, xpix), values)
    img[img==-np.inf] = 0

    mn, mx = img.min(), img.max()
    if mx>mn:
        img = (img-mn)*(255.0/(mx-mn))

    return img.astype(np.uint8), limits, res

# ---------------------
# Zamiana obrazu w współrzędne
# ---------------------
@measure_time
def image_to_coords(img, res, limits, axis="z", threshold=50):
    ys, xs = np.where(img>threshold)
    if len(xs)==0:
        return np.empty((0,2)), ("X","Y")

    if axis=="z":
        x_real = xs*res+limits["x"][0]
        y_real = ys*res+limits["y"][0]
        return np.column_stack((x_real,y_real)), ("X","Y")
    elif axis=="y":
        x_real = xs*res+limits["x"][0]
        z_real = ys*res+limits["z"][0]
        return np.column_stack((x_real,z_real)), ("X","Z")
    else:
        y_real = xs*res+limits["y"][0]
        z_real = ys*res+limits["z"][0]
        return np.column_stack((y_real,z_real)), ("Y","Z")

# ---------------------
# Główna funkcja
# ---------------------
@measure_time
def main(path, axis="z"):
    # Tworzenie katalogu output
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    points = load_point_cloud(path)
    img, limits, res = lidar_to_image(points, axis=axis, res=None, target_size=200)
    print(f"Generated image shape: {img.shape}, resolution={res:.4f} m/pix")

    # Zapis obrazu do katalogu output
    img_path = os.path.join(output_dir, "lidar_image.png")
    cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Obraz zapisano do: {img_path}")

    coords_2d, labels = image_to_coords(img, res, limits, axis=axis)
    print(f"Znaleziono punktów: {len(coords_2d)}\n")

    # Zapis punktów do katalogu output
    out_file = os.path.join(output_dir, "coords_2d.txt")
    header = f"{labels[0]},{labels[1]}\n"
    np.savetxt(out_file, coords_2d, fmt="%.3f", delimiter=",", header=header, comments='')
    print(f"Punkty zapisano do: {out_file}")

    print("Pierwsze punkty:")
    max_show = 10
    for coord in coords_2d[:max_show]:
        print(f"{labels[0]}={coord[0]:.3f}, {labels[1]}={coord[1]:.3f}")
    if len(coords_2d) > max_show:
        print(f"... (pominięto {len(coords_2d)-max_show} punktów)")

# ---------------------
# Entry point
# ---------------------
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: main.py <path> [axis]")
        sys.exit(1)

    path = sys.argv[1]
    axis = "z"
    if len(sys.argv)>=3:
        axis = sys.argv[2].lower()
        if axis not in ["x","y","z"]:
            print("Axis must be one of 'x', 'y', 'z'")
            sys.exit(1)

    main(path, axis)
