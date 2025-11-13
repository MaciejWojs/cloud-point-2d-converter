#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d


def lidar_to_image(
    points, axis="z", res=None, target_size=512, limits=None, value_mode="height"
):
    """
    Tworzy obraz 2D z chmury punktów LiDAR przez rzutowanie na wybraną płaszczyznę.

    Args:
        points (np.ndarray): chmura punktów (N, 3) lub (N, 4) [x, y, z, (intensity)].
        axis (str): oś do 'skompresowania' → 'x', 'y' lub 'z'.
                    - 'z' → rzut na XY (widok z góry)
                    - 'y' → rzut na XZ (widok z boku)
                    - 'x' → rzut na YZ (widok z przodu)
        res (float): rozdzielczość w metrach na piksel (jeśli None, dobierana automatycznie).
        target_size (int): docelowy rozmiar obrazu w pikselach (dla auto-res).
        limits (dict): np. {'x':(-50,50), 'y':(-50,50), 'z':(-5,5)} — zakresy dla osi.
        value_mode (str): 'height' (z), 'intensity' lub 'count'.

    Returns:
        img (np.ndarray): obraz 2D w skali 0–255 (uint8)
        limits (dict): użyte zakresy osi
        res (float): faktyczna rozdzielczość [m/pix]
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
    if axis == "z":  # Bird's-eye view
        plane_axes = ("x", "y")
        value_axis = "z"
    elif axis == "y":  # Widok z boku
        plane_axes = ("x", "z")
        value_axis = "y"
    elif axis == "x":  # Widok z przodu
        plane_axes = ("y", "z")
        value_axis = "x"
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")

    # Ekstrakcja współrzędnych
    coord = {k: points[:, i] for i, k in enumerate(["x", "y", "z"])}
    coord["intensity"] = points[:, 3] if points.shape[1] > 3 else np.ones(len(points))

    # Filtrowanie wg zakresu
    mask = np.ones(len(points), dtype=bool)
    for ax in ["x", "y", "z"]:
        mask &= (coord[ax] >= limits[ax][0]) & (coord[ax] <= limits[ax][1])
    for ax in coord:
        coord[ax] = coord[ax][mask]

    # Zamiana współrzędnych na piksele
    xpix = ((coord[plane_axes[0]] - limits[plane_axes[0]][0]) / res).astype(np.int32)
    ypix = ((coord[plane_axes[1]] - limits[plane_axes[1]][0]) / res).astype(np.int32)

    w = int((limits[plane_axes[0]][1] - limits[plane_axes[0]][0]) / res)
    h = int((limits[plane_axes[1]][1] - limits[plane_axes[1]][0]) / res)
    img = np.zeros((h, w), dtype=np.float32)

    # Wartość piksela (height / intensity / count)
    if value_mode == "height":
        val = coord[value_axis]
    elif value_mode == "intensity":
        val = coord["intensity"]
    elif value_mode == "count":
        val = np.ones_like(coord["x"])
    else:
        raise ValueError("value_mode must be 'height', 'intensity' or 'count'")

    for xi, yi, v in zip(xpix, ypix, val):
        if 0 <= xi < w and 0 <= yi < h:
            img[yi, xi] = max(img[yi, xi], v)

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8), limits, res


def image_to_coords(img, res, limits, axis="z"):
    """
    Zamienia jasne piksele obrazu 2D na współrzędne punktów rzeczywistych.
    """
    ys, xs = np.where(img > 50)  # wybór jasnych pikseli

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


def main(path, axis="z"):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)

    # Rzutowanie
    img, limits, res = lidar_to_image(points, axis=axis, res=None, target_size=200)
    print(f"Generated image shape: {img.shape}, resolution={res:.4f} m/pix")

    cv2.imwrite("lidar_image.png", img)

    # Konwersja obrazu na współrzędne
    coords_2d, labels = image_to_coords(img, res=res, limits=limits, axis=axis)

    for coord in coords_2d:
        print(f"{labels[0]}: {coord[0]:.2f}, {labels[1]}: {coord[1]:.2f}")


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
