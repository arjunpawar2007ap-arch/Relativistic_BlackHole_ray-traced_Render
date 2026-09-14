import os
import numpy as np
import matplotlib.pyplot as plt

res = 400
image = np.zeros((res, res, 3))

bh_position = np.array([0.0, 0.0, 0.0])

r_s = 1.4
disk_inner = 2.5
disk_outer = 6.5

dt = 0.02
max_steps = 900

grid_x, grid_y = np.meshgrid(np.linspace(-9.0, 9.0, res), np.linspace(-9.0, 9.0, res))
num_pixels = res * res

num_frames = 300
fps = 30
os.makedirs("final_frames_2", exist_ok=True)

for frame in range(num_frames):
    image = np.zeros((res, res, 3))
    t = frame / (num_frames - 1)

    yaw = np.radians(-70 + 300 * t)
    pitch = np.radians(-25 + 65 * t + 12 * np.sin(np.pi * t))
    roll = np.radians(8 + 22 * t)
    zoom = 1.15 - 0.35 * np.sin(np.pi * t) + 0.15 * t

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = R_y @ R_x @ R_z

    camera_center = R @ np.array([0.0, 4.5, -14.0])

    local_position = np.zeros((num_pixels, 3))
    local_position[:, 0] = grid_x.flatten() * zoom
    local_position[:, 1] = grid_y.flatten() * zoom

    position = local_position @ R.T + camera_center

    velocity = np.zeros((num_pixels, 3))
    velocity[:, 1] = -0.28
    velocity[:, 2] = 1.0
    velocity = velocity @ R.T

    velocity /= np.sqrt(np.sum(velocity**2, axis=1, keepdims=True))

    disk_glow = np.zeros(num_pixels)
    active_rays = np.ones(num_pixels, dtype=bool)

    for step in range(max_steps):
        if not np.any(active_rays):
            break

        r_vectors = position - bh_position
        r_mags = np.sqrt(np.sum(r_vectors**2, axis=1))

        captured = (r_mags <= r_s) & active_rays
        active_rays[captured] = False
        velocity[captured] = 0.0

        ang_mom_sq = np.sum(np.cross(r_vectors, velocity)**2, axis=1)
        force_mag = -1.5 * r_s * ang_mom_sq / (r_mags**5)
        acceleration = force_mag[:, np.newaxis] * r_vectors

        velocity[active_rays] += acceleration[active_rays] * dt

        old_y = position[:, 1].copy()
        position[active_rays] += velocity[active_rays] * dt

        crossed_equator = active_rays & (((old_y > 0) & (position[:, 1] <= 0)) | ((old_y < 0) & (position[:, 1] >= 0)))

        if np.any(crossed_equator):
            cross_r = np.sqrt(position[crossed_equator, 0]**2 + position[crossed_equator, 2]**2)
            hit_disk = (cross_r >= disk_inner) & (cross_r <= disk_outer)

            if np.any(hit_disk):
                crossed_indices = np.where(crossed_equator)[0]
                hit_indices = crossed_indices[hit_disk]
                intensities = (disk_outer - cross_r[hit_disk]) / (disk_outer - disk_inner)

                disk_glow[hit_indices] = intensities
                active_rays[hit_indices] = False
                velocity[hit_indices] = 0.0

    disk_glow = disk_glow.reshape((res, res))

    r_final_mags = np.sqrt(np.sum((position - bh_position)**2, axis=1)).reshape((res, res))
    horizon_mask = r_final_mags <= r_s + 0.03
    glow_mask = (disk_glow > 0) & (~horizon_mask)

    image[glow_mask, 0] = disk_glow[glow_mask]**0.3
    image[glow_mask, 1] = disk_glow[glow_mask]**0.9 * 0.65
    image[glow_mask, 2] = disk_glow[glow_mask]**2 * 0.15
    image[~glow_mask & ~horizon_mask] = [0.003, 0.003, 0.008]

    plt.imsave(f"final_frames_2/frame_{frame:05d}.png", image)
    plt.close()