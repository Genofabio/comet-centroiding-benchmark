import numpy as np
import cv2
from enum import Enum
from scipy.optimize import least_squares

class ClassicAlgorithmType(Enum):
    GLOBAL_LUMINOUS_BARYCENTER = "Global Luminous Barycenter"
    ADAPTIVE_STATISTICAL_THRESHOLD_BARYCENTER = "Adaptive Statistical Threshold Barycenter"
    LOCAL_PARABOLIC_INTERPOLATION = "Local Parabolic Interpolation"
    INTERSECTION_OF_RADIAL_GRADIENTS = "Intersection of Radial Gradients"
    MOVING_WINDOW_SYMMETRY_MAPPING = "Moving Window Symmetry Mapping"
    NUCLEUS_COMA_DECOMPOSITION = "Nucleus-Coma Decomposition"


# ==============================================================================
# GLOBAL LUMINOUS BARYCENTER
# ==============================================================================
def find_center_global_luminous_barycenter(roi, x_off, y_off):
    m = cv2.moments(roi.astype(np.float64))
    if m["m00"] == 0: 
        return float('inf'), float('inf'), float('inf')
    
    xc_local = m["m10"] / m["m00"]
    yc_local = m["m01"] / m["m00"]
    
    return x_off + xc_local, y_off + yc_local, 0.0


# ==============================================================================
# ADAPTIVE STATISTICAL THRESHOLD BARYCENTER
# ==============================================================================
def find_center_adaptive_statistical_threshold_barycenter(roi, x_off, y_off):
    mean, std = np.mean(roi), np.std(roi)
    threshold = mean + (std * 3.0)
    
    masked_roi = np.where(roi > threshold, roi, 0.0)
    m = cv2.moments(masked_roi.astype(np.float64))
    
    if m["m00"] == 0: 
        return find_center_global_luminous_barycenter(roi, x_off, y_off)
    
    xc_local = m["m10"] / m["m00"]
    yc_local = m["m01"] / m["m00"]
    
    return x_off + xc_local, y_off + yc_local, 0.0


# ==============================================================================
# LOCAL PARABOLIC INTERPOLATION
# ==============================================================================
def find_center_local_parabolic_interpolation(roi, x_off, y_off):
    _, _, _, max_loc = cv2.minMaxLoc(roi)
    x0, y0 = max_loc
    
    if 0 < x0 < roi.shape[1]-1 and 0 < y0 < roi.shape[0]-1:
        c = roi[y0, x0]
        l, r = roi[y0, x0-1], roi[y0, x0+1]
        u, d = roi[y0-1, x0], roi[y0+1, x0]
        
        dxx = r + l - 2*c
        dyy = d + u - 2*c
        
        sub_x = x0 - (r - l) / (2.0 * dxx) if abs(dxx) > 1e-9 else x0
        sub_y = y0 - (d - u) / (2.0 * dyy) if abs(dyy) > 1e-9 else y0
        
        return x_off + sub_x, y_off + sub_y, 0.0
    
    return x_off + x0, y_off + y0, 0.0


# ==============================================================================
# INTERSECTION OF RADIAL GRADIENTS
# ==============================================================================
def find_center_intersection_of_radial_gradients(roi, x_off, y_off):
    gx = cv2.Sobel(roi.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    
    mag_sq = gx**2 + gy**2
    weight = np.sqrt(mag_sq)
    
    h, w = roi.shape
    y_idx, x_idx = np.indices((h, w))
    
    A11 = np.sum(weight * gy**2)
    A12 = np.sum(weight * -gx * gy)
    A22 = np.sum(weight * gx**2)
    
    B1 = np.sum(weight * gy * (gy * (x_idx + 0.5) - gx * (y_idx + 0.5)))
    B2 = np.sum(weight * -gx * (gy * (x_idx + 0.5) - gx * (y_idx + 0.5)))
    
    det = A11 * A22 - A12**2
    if abs(det) < 1e-12: 
        return float('inf'), float('inf'), float('inf')
    
    xc_local = (B1 * A22 - A12 * B2) / det
    yc_local = (A11 * B2 - B1 * A12) / det
    
    return x_off + xc_local, y_off + yc_local, 0.0


# ==============================================================================
# MOVING WINDOW SYMMETRY MAPPING
# ==============================================================================
def find_center_moving_window_symmetry_mapping(roi, x_off, y_off, sieve_size=5):
    h, w = roi.shape
    offset = sieve_size // 2
    weighted_x, weighted_y, total_w = 0.0, 0.0, 0.0
    
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            sieve = roi[y-offset:y+offset+1, x-offset:x+offset+1]
            magnitude = np.sum(sieve)
            
            if magnitude <= 1e-9: continue
            
            sieve_rot = np.rot90(sieve, 2)
            diff = np.sum(np.abs(sieve - sieve_rot))
            
            symmetry = max(0, 1.0 - (diff / magnitude))
            weight = magnitude * (symmetry**2)
            
            weighted_x += (x + 0.5) * weight
            weighted_y += (y + 0.5) * weight
            total_w += weight
            
    if total_w <= 1e-9: 
        return float('inf'), float('inf'), float('inf')
    
    xc, yc = weighted_x / total_w, weighted_y / total_w
    return x_off + xc, y_off + yc, 0.0


# ==============================================================================
# NUCLEUS-COMA DECOMPOSITION
# ==============================================================================
def find_center_nucleus_coma_decomposition(roi, x_off, y_off):
    h, w = roi.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    bg = np.percentile(roi, 5)
    psf_sigma = 1.5
    
    def model_nucleus_coma(p, x, y):
        rho = np.sqrt((x - p[2])**2 + (y - p[3])**2)
        nucleus = np.exp(-0.5 * (rho**2) / (psf_sigma**2))
        coma = 1.0 / np.sqrt(rho**2 + psf_sigma**2)
        return p[4] + (p[0] * nucleus) + (p[1] * coma)
    
    p0 = [np.max(roi)/2, np.max(roi)/2, w/2, h/2, bg]
    res = least_squares(lambda p, x, y, z: (model_nucleus_coma(p, x, y) - z).ravel(), p0,
                        args=(x_grid, y_grid, roi), ftol=1e-4)
    
    return x_off + res.x[2], y_off + res.x[3], np.sqrt(np.mean(res.fun**2))