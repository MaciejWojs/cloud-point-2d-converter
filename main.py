#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps, lru_cache
import os
import laspy


def measure_time(func):
    """Dekorator mierzący czas wykonywania funkcji."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[BENCHMARK]  {func.__name__} wykonał się w {elapsed:.4f} sekundach")
        return result
    return wrapper


@measure_time
def lidar_to_image(
    points, axis="z", res=None, target_size=512, limits=None, value_mode="height"
):
    """
    Tworzy obraz 2D z chmury punktów LiDAR przez rzutowanie na wybraną płaszczyznę.
    Zoptymalizowana wersja z operacjami wektorowymi.
    """

    # Automatyczne wyznaczenie zakresów, jeśli brak
    if limits is None:
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        limits = {
            "x": (mins[0], maxs[0]),
            "y": (mins[1], maxs[1]),
            "z": (mins[2], maxs[2]),
        }

    # Automatyczny dobór rozdzielczości
    if res is None:
        max_range = max(
            limits["x"][1] - limits["x"][0],
            limits["y"][1] - limits["y"][0],
            limits["z"][1] - limits["z"][0],
        )
        res = max_range / target_size

    # Wybór osi rzutowania
    if axis == "z":
        plane_axes = ("x", "y")
        value_axis = "z"
        plane_idx = (0, 1)
        value_idx = 2
    elif axis == "y":
        plane_axes = ("x", "z")
        value_axis = "y"
        plane_idx = (0, 2)
        value_idx = 1
    elif axis == "x":
        plane_axes = ("y", "z")
        value_axis = "x"
        plane_idx = (1, 2)
        value_idx = 0
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")

    # Ekstrakcja współrzędnych (wektorowo)
    pts_3d = points[:, :3]
    intensity = points[:, 3] if points.shape[1] > 3 else np.ones(len(points))

    # Filtrowanie wg zakresu (wektorowo)
    mask = np.ones(len(points), dtype=bool)
    limits_arr = np.array([limits["x"], limits["y"], limits["z"]])
    for i in range(3):
        mask &= (pts_3d[:, i] >= limits_arr[i, 0]) & (pts_3d[:, i] <= limits_arr[i, 1])
    
    pts_filtered = pts_3d[mask]
    intensity_filtered = intensity[mask]

    # Zamiana współrzędnych na piksele (wektorowo)
    plane_coords = pts_filtered[:, plane_idx]
    plane_limits = np.array([limits[plane_axes[0]], limits[plane_axes[1]]])
    
    xpix = ((plane_coords[:, 0] - plane_limits[0, 0]) / res).astype(np.int32)
    ypix = ((plane_coords[:, 1] - plane_limits[1, 0]) / res).astype(np.int32)

    w = int((plane_limits[0, 1] - plane_limits[0, 0]) / res)
    h = int((plane_limits[1, 1] - plane_limits[1, 0]) / res)
    img = np.zeros((h, w), dtype=np.float32)

    # Wartość piksela
    if value_mode == "height":
        val = pts_filtered[:, value_idx]
    elif value_mode == "intensity":
        val = intensity_filtered
    elif value_mode == "count":
        val = np.ones(len(pts_filtered))
    else:
        raise ValueError("value_mode must be 'height', 'intensity' or 'count'")

    # Wektorowa agregacja pikseli zamiast pętli
    valid_mask = (xpix >= 0) & (xpix < w) & (ypix >= 0) & (ypix < h)
    xpix_valid = xpix[valid_mask]
    ypix_valid = ypix[valid_mask]
    val_valid = val[valid_mask]
    
    # Konwersja 2D indeksów na 1D
    indices = ypix_valid * w + xpix_valid
    np.maximum.at(img.ravel(), indices, val_valid)

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8), limits, res


@measure_time
def image_to_coords(img, res, limits, axis="z"):
    """
    Zamienia jasne piksele obrazu 2D na współrzędne punktów rzeczywistych.
    Zoptymalizowana wersja.
    """
    ys, xs = np.where(img > 50)

    if axis == "z":
        x_real = xs * res + limits["x"][0]
        y_real = ys * res + limits["y"][0]
        return np.vstack((x_real, y_real)).T, ("X", "Y")
    elif axis == "y":
        x_real = xs * res + limits["x"][0]
        z_real = ys * res + limits["z"][0]
        return np.vstack((x_real, z_real)).T, ("X", "Z")
    elif axis == "x":
        y_real = xs * res + limits["y"][0]
        z_real = ys * res + limits["z"][0]
        return np.vstack((y_real, z_real)).T, ("Y", "Z")


@lru_cache(maxsize=8)
@measure_time
def load_point_cloud(path):
    """
    Ładuje chmurę punktów z cache'em dla powtórnych wczytań.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".obj", ".off"]:
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        # Obsługa 2D - dodaj kolumnę Z jeśli brakuje
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((len(points), 1))])
        if pcd.has_colors():
            intensity = np.mean(np.asarray(pcd.colors), axis=1)
            points = np.hstack([points, intensity.reshape(-1, 1)])
        return points
    elif ext in [".las", ".laz"]:
        las = laspy.read(path)
        points = np.vstack((las.x, las.y, las.z)).T
        if hasattr(las, "intensity"):
            intensity = las.intensity
            points = np.hstack([points, intensity.reshape(-1, 1)])
        return points
    elif ext in [".csv", ".txt"]:
        data = np.loadtxt(path, delimiter=",")
        return data
    else:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")


@measure_time
def main(path, axis="z"):
    points = load_point_cloud(path)

    # Rzutowanie
    img, limits, res = lidar_to_image(points, axis=axis, res=None, target_size=200)
    print(f"Generated image shape: {img.shape}, resolution={res:.4f} m/pix")

    cv2.imwrite("lidar_image.png", img)

    # Konwersja obrazu na współrzędne
    coords_2d, labels = image_to_coords(img, res=res, limits=limits, axis=axis)

    # Zapis punktów do pliku tekstowego
    out_file = "coords_2d.txt"
    header = f"{labels[0]},{labels[1]}\n"
    np.savetxt(out_file, coords_2d, fmt="%.6f", delimiter=",", header=header, comments='')
    print(f"Punkty zapisano do pliku: {out_file}")

    # Opcjonalnie: wyświetlanie kilku pierwszych punktów
    max_show = 10
    for coord in coords_2d[:max_show]:
        print(f"{labels[0]}: {coord[0]:.2f}, {labels[1]}: {coord[1]:.2f}")
    if len(coords_2d) > max_show:
        print(f"... (pominięto {len(coords_2d) - max_show} punktów)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <path> [axis]")
        sys.exit(1)

    path = sys.argv[1]
    axis = "z"
    if len(sys.argv) >= 3:
        axis = sys.argv[2]
        axis = axis.lower()
        if axis not in ["x", "y", "z"]:
            print("Axis must be one of 'x', 'y', or 'z'.")
            sys.exit(1)

    main(path, axis)
