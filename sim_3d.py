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

position = np.zeros((num_pixels, 3))
position[:, 0] = grid_x.flatten()
position[:, 1] = grid_y.flatten() + 4.5
position[:, 2] = -14.0                  

velocity = np.zeros((num_pixels, 3))
velocity[:, 0] = grid_x.flatten() / 14.0
velocity[:, 1] = (grid_y.flatten() / 14.0) - 0.28
velocity[:, 2] = 1.0
v_mags = np.sqrt(np.sum(velocity**2, axis=1, keepdims=True))
velocity /= v_mags

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
    
    crossed_equator = active_rays & ((old_y > 0) & (position[:, 1] <= 0) | (old_y < 0) & (position[:, 1] >= 0))
    
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

plt.figure(figsize=(10, 10))
plt.imshow(image, extent=[-7, 7, -7, 7], origin='lower')
plt.axis('off')
plt.gcf().patch.set_facecolor('black')
plt.show()
