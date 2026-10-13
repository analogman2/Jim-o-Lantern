#!/usr/bin/env python3
"""Render preview PNGs of the case: an exploded assembly plus each printed part."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build123d import Pos, Rot, export_stl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

import jim_case as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def tris(part, tmp):
    export_stl(part, tmp, tolerance=0.2, angular_tolerance=0.3)
    return mesh.Mesh.from_file(tmp).vectors


def draw(ax, vectors, color, alpha=1.0):
    n = np.cross(vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
    shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    ax.add_collection3d(Poly3DCollection(vectors, facecolors=shade[:, None] * base,
                                         edgecolor="none", alpha=alpha))
    return vectors.reshape(-1, 3)


def finish(ax, pts, elev, azim, title, path):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()
    print("wrote", path)


def main():
    tmp = os.path.join(OUT, "_tmp.stl")
    # exploded assembly: lid lifted and flipped right side up, socket below, cup and camera inside
    lid = Pos(-J.LID_OVERHANG, J.D + J.LID_OVERHANG, J.H + J.LID_T + 45) * Rot(X=180) * J.lid()
    cup = Pos(J.SPK_X, J.WALL + 8, J.SPK_Z) * Rot(X=-90) * J.speaker_cup()
    cam = Pos(J.CAM_X, J.WALL + 7, J.CAM_Z) * Rot(X=-90) * J.camera_mount()
    sock = Pos(J.W / 2, J.SOCKET_Y, -60) * J.post_socket()
    fig = plt.figure(figsize=(9, 9)); ax = fig.add_subplot(projection="3d")
    pts = [draw(ax, tris(J.body(), tmp), "#e07a1f"),
           draw(ax, tris(lid, tmp), "#3a3a3a", 0.85),
           draw(ax, tris(cup, tmp), "#6b6b6b"),
           draw(ax, tris(cam, tmp), "#2a7fd4"),
           draw(ax, tris(sock, tmp), "#6b6b6b")]
    finish(ax, np.vstack(pts), 22, -62, "Jim's box: exploded view", os.path.join(OUT, "preview_assembly.png"))

    for name, make, color, elev, azim in [
            ("body", J.body, "#e07a1f", 25, -60), ("body_inside", J.body, "#e07a1f", 60, 125),
            ("lid", J.lid, "#3a3a3a", 50, -60), ("speaker_cup", J.speaker_cup, "#6b6b6b", 30, -60),
            ("camera_mount", J.camera_mount, "#2a7fd4", 45, -60),
            ("post_socket", J.post_socket, "#6b6b6b", -25, -60)]:
        fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(projection="3d")
        p = draw(ax, tris(make(), tmp), color)
        finish(ax, p, elev, azim, name.replace("_", " "), os.path.join(OUT, f"preview_{name}.png"))
    os.remove(tmp)


if __name__ == "__main__":
    main()
