import numpy as np
import cv2
import traceback
from enum import Enum
from scipy.optimize import least_squares, minimize

class AlgorithmType(Enum):
    TWO_DIMENSIONAL_GAUSSIAN_FITTING = "Two-Dimensional Gaussian Fitting"
    ONE_DIMENSIONAL_POWER_LAW_PROFILE = "One-Dimensional Power-Law Profile"
    TRUNCATED_RADIAL_POWER_LAW_PROFILE = "Truncated Radial Power-Law Profile"
    CENTRAL_SOFTENED_RADIAL_PROFILE = "Central Softened Radial Profile"
    SOFTENED_AND_ROTATABLE_ELLIPTICAL_PROFILE = "Softened and Rotatable Elliptical Profile"
    ASYMMETRIC_ROTATIONAL_HYBRID_PROFILE = "Asymmetric-Rotational Hybrid Profile"
    BOUNDED_SOFTENED_RADIAL_PROFILE = "Bounded Softened Radial Profile"
    ANISOTROPIC_ORTHOGONAL_PROFILE = "Anisotropic Orthogonal Profile"
    ASYMMETRIC_QUADRANT_PROFILE = "Asymmetric Quadrant Profile"
    ASYMMETRIC_WEIGHTED_RESIDUAL_PROFILE = "Asymmetric Weighted Residual Profile"
    CSHARP_CLONE = "CSharp_LBFGSB_Clone"


# ==============================================================================
# TWO-DIMENSIONAL GAUSSIAN FITTING
# ==============================================================================
def two_dimensional_gaussian_residuals(params, x, y, data_z):
    xc, yc, sx, sy, theta, amp, offset = params
    cost2, sint2 = np.cos(theta)**2, np.sin(theta)**2
    sin2t = np.sin(2*theta)
    x_std, y_std = x - xc, y - yc
    a = ((cost2/(2*sx**2)) + (sint2/(2*sy**2))) * x_std**2
    b = ((sint2/(2*sx**2)) + (cost2/(2*sy**2))) * y_std**2
    c = ((sin2t/(4*sx**2)) - (sin2t/(4*sy**2))) * x_std * y_std
    model = amp * np.exp(-(a + b + 2*c)) + offset
    return (model - data_z).ravel()

def find_center_two_dimensional_gaussian_fitting(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, 2.0, 2.0, 0.0, np.max(roi)-bg, bg]
        res = least_squares(two_dimensional_gaussian_residuals, p0, args=(x_grid, y_grid, roi),
                            bounds=([0,0,0.1,0.1,-np.pi/2,0,-np.inf],[w,h,20,20,np.pi/2,np.inf,np.inf]), 
                            loss='soft_l1', ftol=1e-4)
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# ONE-DIMENSIONAL POWER-LAW PROFILE
# ==============================================================================
def one_dimensional_power_law_model(params, x_coords, sub_samples=10):
    delta, A, n_neg, n_pos, a, bg = params
    offsets = np.linspace(-0.5, 0.5, sub_samples)
    model_values = []
    for x_i in x_coords:
        x_sub = (x_i + offsets) - delta
        n = np.where(x_sub < 0, n_neg, n_pos)
        flux = A / (np.abs(x_sub)**n + a)
        model_values.append(np.mean(flux) + bg)
    return np.array(model_values)

def find_center_one_dimensional_power_law_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(roi)
        px_loc, py_loc = max_l
        prof_x, prof_y = roi[py_loc, :], roi[:, px_loc]
        bg = np.percentile(roi, 5)
        amp = max_v - bg
        res_x = least_squares(lambda p, x, y: one_dimensional_power_law_model(p, x) - y, 
                              [px_loc, amp, 1.0, 1.0, 1.5, bg], args=(np.arange(w), prof_x),
                              bounds=([0,0,0.1,0.1,0.1,-np.inf],[w,np.inf,5,5,20,np.inf]), ftol=1e-4)
        res_y = least_squares(lambda p, x, y: one_dimensional_power_law_model(p, x) - y, 
                              [py_loc, amp, 1.0, 1.0, 1.5, bg], args=(np.arange(h), prof_y),
                              bounds=([0,0,0.1,0.1,0.1,-np.inf],[h,np.inf,5,5,20,np.inf]), ftol=1e-4)
        total_rms = np.sqrt((np.mean(res_x.fun**2) + np.mean(res_y.fun**2))/2)
        return x_off + res_x.x[0], y_off + res_y.x[0], total_rms
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# TRUNCATED RADIAL POWER-LAW PROFILE
# ==============================================================================
def find_center_truncated_radial_power_law_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, (np.max(roi)-bg)*0.5, 1.0, bg]
        def model_truncated_radial(p, x, y):
            dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
            return p[2] * (np.maximum(dist, 0.5)**(-p[3])) + p[4]
        res = least_squares(lambda p, x, y, z: (model_truncated_radial(p, x, y) - z).ravel(),
                            p0, args=(x_grid, y_grid, roi), 
                            bounds=([0,0,1e-5,0.1,-np.inf],[w,h,np.inf,4,np.inf]), ftol=1e-5)
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# CENTRAL SOFTENED RADIAL PROFILE
# ==============================================================================
def find_center_central_softened_radial_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, np.max(roi)-bg, 1.0, 1.5, bg]
        def model_central_softened(p, x, y):
            r_sq = (x-p[0])**2 + (y-p[1])**2
            return p[2] * (r_sq + p[4]**2)**(-p[3]/2.0) + p[5]
        res = least_squares(lambda p, x, y, z: (model_central_softened(p, x, y) - z).ravel(),
                            p0, args=(x_grid, y_grid, roi), 
                            bounds=([0,0,0,0.1,0.1,-np.inf],[w,h,np.inf,5,20,np.inf]), ftol=1e-5)
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# SOFTENED AND ROTATABLE ELLIPTICAL PROFILE
# ==============================================================================
def find_center_softened_and_rotatable_elliptical_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        
        p0 = [w/2.0, h/2.0, np.max(roi)-bg, 1.0, 1.5, bg, 1.0, 0.0]
        
        def model_softened_rotated(p, x, y):
            dx, dy = x - p[0], y - p[1]
            xr = dx * np.cos(p[7]) + dy * np.sin(p[7])
            yr = -dx * np.sin(p[7]) + dy * np.cos(p[7])
            return p[2] * ( (xr**2 + (yr/p[6])**2) + p[4]**2 )**(-p[3]/2.0) + p[5]
        
        b_lower = [0, 0, 0.0, 0.1, 0.1, -np.inf, 0.1, -np.pi/2]
        b_upper = [w, h, np.inf, 5.0, 20.0, np.inf, 10.0, np.pi/2]
        
        res = least_squares(lambda p, x, y, z: (model_softened_rotated(p, x, y) - z).ravel(), p0, args=(x_grid, y_grid, roi),
                            bounds=(b_lower, b_upper), loss='soft_l1')
        
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except Exception as e:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# ASYMMETRIC-ROTATIONAL HYBRID PROFILE
# ==============================================================================
def find_center_asymmetric_rotational_hybrid_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, np.max(roi)-bg, 1.5, bg, 0.0, 1.0, 1.0]
        def model_hybrid(p, x, y):
            dx, dy = x - p[0], y - p[1]
            xr = dx * np.cos(p[5]) + dy * np.sin(p[5])
            yr = -dx * np.sin(p[5]) + dy * np.cos(p[5])
            nm = np.where(xr >= 0, p[6], p[7])
            return p[2] * ( (xr**2 + yr**2) + p[3]**2 )**(-nm/2.0) + p[4]
        res = least_squares(lambda p, x, y, z: (model_hybrid(p, x, y) - z).ravel(), p0, args=(x_grid, y_grid, roi),
                            bounds=([0,0,0,0.1,-np.inf,-np.pi/2,0.1,0.1],[w,h,np.inf,20,np.inf,np.pi/2,5,5]), loss='soft_l1')
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# BOUNDED SOFTENED RADIAL PROFILE
# ==============================================================================
def find_center_bounded_softened_radial_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, np.max(roi)-bg, 1.0, 1.5, bg]
        def model_bounded_softened(p, x, y):
            return p[2] * ((x-p[0])**2 + (y-p[1])**2 + p[4]**2)**(-p[3]/2.0) + p[5]
        res = least_squares(lambda p, x, y, z: (model_bounded_softened(p, x, y) - z).ravel(), p0, args=(x_grid, y_grid, roi), 
                            bounds=([0,0,0,0.1,0.1,-np.inf],[w,h,np.inf,4,15,np.inf]), ftol=1e-5)
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# ANISOTROPIC ORTHOGONAL PROFILE
# ==============================================================================
def find_center_anisotropic_orthogonal_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        p0 = [w/2.0, h/2.0, (np.max(roi)-bg)*2.0, 1.0, 1.0, 1.5, bg]
        def model_anisotropic(p, x, y):
            term_x = ((x-p[0])**2 + p[5]**2)**(p[3]/2.0)
            term_y = ((y-p[1])**2 + p[5]**2)**(p[4]/2.0)
            return p[2] * (1.0 / (term_x + term_y)) + p[6]
        res = least_squares(lambda p, x, y, z: (model_anisotropic(p, x, y) - z).ravel(), p0, args=(x_grid, y_grid, roi),
                            bounds=([0,0,0,0.1,0.1,0.1,-np.inf],[w,h,np.inf,4,4,15,np.inf]), ftol=1e-5)
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# ASYMMETRIC QUADRANT PROFILE
# ==============================================================================
def find_center_asymmetric_quadrant_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        max_val = np.max(roi)
        
        start_x, start_y = w / 2.0, h / 2.0
        amp = (max_val - bg) * 1.5 
        
        p0 = [start_x, start_y, amp, 1.0, 1.0, 1.0, 1.0, 1.5, bg]
        
        def model_cusp_asym(p, x, y):
            xc, yc, A, nxP, nxM, nyP, nyM, r0, bgVal = p
            dx = x - xc
            dy = y - yc
            
            nx = np.where(dx >= 0, nxP, nxM)
            ny = np.where(dy >= 0, nyP, nyM)
            
            term_x = np.abs(dx)**nx
            term_y = np.abs(dy)**ny
            
            return (A / (term_x + term_y + r0)) + bgVal

        res = least_squares(lambda p, x, y, z: (model_cusp_asym(p, x, y) - z).ravel(), p0, 
                            args=(x_grid, y_grid, roi),
                            bounds=([0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, -1e6],
                                    [w, h, 1e9, 5.0, 5.0, 5.0, 5.0, 20.0, 1e9]), 
                            ftol=1e-5)
        
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# CSHARP CLONE OF ASYMMETRIC QUADRANT PROFILE
# ==============================================================================
def find_center_asymmetric_quadrant_profile_csharp_clone(roi, x_off=0, y_off=0):
    try:
        h, w = roi.shape
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        bg = np.percentile(roi, 5)
        max_val = np.max(roi)
        
        start_x, start_y = w / 2.0, h / 2.0
        amp = (max_val - bg) * 1.5 
        
        p0 = [start_x, start_y, amp, 1.0, 1.0, 1.0, 1.0, 1.5, bg]
        
        def model_cusp_asym_clone(p, x, y):
            xc, yc, A, nxP, nxM, nyP, nyM, r0, bgVal = p
            dx = x - xc
            dy = y - yc
            
            nx = np.where(dx >= 0, nxP, nxM)
            ny = np.where(dy >= 0, nyP, nyM)
            
            term_x = np.abs(dx)**nx
            term_y = np.abs(dy)**ny
            
            return (A / (term_x + term_y + r0)) + bgVal

        res = least_squares(lambda p, x, y, z: (model_cusp_asym_clone(p, x, y) - z).ravel(), p0, 
                            args=(x_grid, y_grid, roi),
                            bounds=([0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, -1e6],
                                    [w, h, 1e9, 5.0, 5.0, 5.0, 5.0, 20.0, 1e9]), 
                            ftol=1e-5)
        
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')


# ==============================================================================
# ASYMMETRIC WEIGHTED RESIDUAL PROFILE
# ==============================================================================
def find_center_asymmetric_weighted_residual_profile(roi, x_off, y_off):
    try:
        h, w = roi.shape
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        bg = np.percentile(roi, 10)
        amp = np.max(roi) - bg
        
        sigma_w = min(h, w) / 3.5 
        weight = np.exp(-((x_grid - w/2)**2 + (y_grid - h/2)**2) / (2 * sigma_w**2))
        
        p0 = [w/2, h/2, amp, 1.0, 1.0, 1.0, 1.0, 1.5, bg]
        
        def model_asym_weighted(p, x, y):
            dx, dy = x - p[0], y - p[1]
            nx = np.where(dx >= 0, p[3], p[4])
            ny = np.where(dy >= 0, p[5], p[6])
            term_x = (dx**2 + p[7]**2)**(nx / 2.0)
            term_y = (dy**2 + p[7]**2)**(ny / 2.0)
            return p[2] * (1.0 / (term_x + term_y)) + p[8]

        def resid_weighted(p):
            return ((model_asym_weighted(p, x_grid, y_grid) - roi) * weight).ravel()

        res = least_squares(resid_weighted, p0, 
                            bounds=([0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.5, -np.inf],
                                    [w, h, np.inf, 5.0, 5.0, 5.0, 5.0, 10.0, np.inf]),
                            loss='soft_l1', f_scale=amp*0.1, ftol=1e-5)
        
        return x_off + res.x[0], y_off + res.x[1], np.sqrt(np.mean(res.fun**2))
    except:
        return float('inf'), float('inf'), float('inf')