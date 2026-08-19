import os
import csv
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from datetime import datetime, timedelta

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
OUTPUT_DIR = "data/synthetic_test"
TRUTH_FILE = "data/synthetic_ground_truth.csv"
NUM_IMAGES = 300
IMG_SIZE = 400


# =============================================================================
# PARAMETER CONFIGURATION FUNCTIONS
# =============================================================================

def get_params_baseline():
    return {
        "cat_name": "Baseline",
        "sky_bg_base": np.random.uniform(9500.0, 12500.0),
        "has_dark_blob": np.random.choice([True, False]),
        "dark_blob_dip": np.random.uniform(-800.0, -100.0),
        "dark_blob_sigma": np.random.uniform(30.0, 100.0),
        "dark_blob_aspect": np.random.uniform(1.2, 2.0),
        "corner_boost1": np.random.uniform(100.0, 600.0),
        "corner_boost2": np.random.uniform(50.0, 400.0),
        "sky_mosaic_amplitude": np.random.uniform(5.0, 40.0),
        "noise_std": np.random.uniform(15.0, 40.0),
        "target_peak_above_bg": np.random.uniform(90000.0, 145000.0),
        "nucleus_fwhm": np.random.uniform(2.0, 4.5),               
        "theta_sun": np.random.uniform(0, 2 * np.pi),             
        "jet_opening_angle": np.random.uniform(10.0, 30.0),       
        "jet_peak_distance": np.random.uniform(0.0, 0.008),        
        "drift_rate": np.random.uniform(0.00, 0.01),              
        "coma_aspect_ratio": np.random.uniform(1.01, 1.06),       
        "fade_radius_global": np.random.uniform(70.0, 220.0),     
        "star_classes": [
            {
                "count": np.random.randint(1, 5),
                "amp": (75000.0, 130000.0),
                "alpha_range": (1.6, 2.0),
                "beta_range": (1.5, 1.8)
            },
            {
                "count": np.random.randint(5, 15),
                "amp": (3500.0, 7500.0),
                "alpha_range": (1.4, 1.8),
                "beta_range": (1.3, 1.5)
            },
            {
                "count": np.random.randint(5, 20),
                "amp": (600.0, 1600.0),
                "alpha_range": (1.3, 1.8),
                "beta_range": (1.05, 1.3)
            },
        ],
        "num_hot_pixels": np.random.randint(0, 7),
        "num_dead_pixels": np.random.randint(0, 7),
    }


def get_params_low_snr():
    return {
        "cat_name": "Low_SNR",
        "sky_bg_base": np.random.uniform(1800.0, 3600.0),            
        "noise_std": np.random.uniform(35.0, 80.0),
        "sky_mosaic_amplitude": np.random.uniform(3.0, 15.0),
        "target_peak_above_bg": np.random.uniform(1200.0, 4000.0),   
        "nucleus_fwhm": np.random.uniform(1.0, 2.5),
        "fade_radius_global": np.random.uniform(2.0, 7.0),          
        "theta_sun": np.random.uniform(0, 2 * np.pi),
        "drift_rate": np.random.uniform(0.005, 0.06),                
        "coma_aspect_ratio": np.random.uniform(1.01, 1.05),          
        "star_classes": [
            {
                "count": np.random.randint(5, 16),                    
                "amp": (35000.0, 75000.0),
                "alpha_range": (0.9, 1.6),
                "beta_range": (1.4, 2.3)
            },
            {
                "count": np.random.randint(20, 45),                 
                "amp": (4500.0, 11000.0),
                "alpha_range": (0.8, 1.7),
                "beta_range": (1.25, 1.85)
            },
            {
                "count": np.random.randint(40, 80),                 
                "amp": (3000.0, 8500.0),
                "alpha_range": (0.7, 1.5),
                "beta_range": (1.30, 1.95)
            },
            {
                "count": np.random.randint(80, 160),                  
                "amp": (900.0, 3200.0),
                "alpha_range": (0.6, 1.4),
                "beta_range": (1.35, 2.10)
            },
            {
                "count": np.random.randint(120, 300),                
                "amp": (150.0, 600.0),
                "alpha_range": (0.5, 1.2),
                "beta_range": (1.40, 2.30)
            },
            {
                "count": np.random.randint(150, 500),                
                "amp": (30.0, 350.0),
                "alpha_range": (0.8, 2.1),
                "beta_range": (1.15, 1.65)
            },
        ],
        "num_hot_pixels": np.random.randint(1, 10),
        "num_dead_pixels": np.random.randint(1, 7),
    }


def get_params_asymmetric():
    return {
        "cat_name": "Asymmetric",
        "sky_bg_base": np.random.uniform(500.0, 850.0),             
        "noise_std": np.random.uniform(8.0, 25.0),                  
        "sky_mosaic_amplitude": np.random.uniform(1.5, 8.5),
        "target_peak_above_bg": np.random.uniform(1800.0, 3200.0),  
        "nucleus_fwhm": np.random.uniform(1.8, 3.2),              
        "core_oval_ratio": np.random.uniform(1.06, 1.12),          
        "theta_sun": np.random.uniform(0, 2 * np.pi),              
        "fade_radius_base": np.random.uniform(25.0, 60.0),         
        "tail_extension_ratio": np.random.uniform(6.0, 9.0),      
        "transition_width": 12.0,                                   
        "star_classes": [
            {
                "count": np.random.randint(1, 5),
                "amp": (6500.0, 65000.0),
                "alpha_range": (1.0, 1.6),
                "beta_range": (1.3, 1.8)
            },
            {
                "count": np.random.randint(5, 18),
                "amp": (2500.0, 7000.0),
                "alpha_range": (1.0, 2.0),
                "beta_range": (1.2, 1.7)
            },
            {
                "count": np.random.randint(10, 35),
                "amp": (150.0, 2000.0),
                "alpha_range": (0.8, 1.8),
                "beta_range": (1.25, 1.85)
            },
        ],
        "num_hot_pixels": np.random.randint(1, 8),
        "num_dead_pixels": np.random.randint(0, 6),
    }


# =============================================================================
# IMAGE RENDERING PIPELINES
# =============================================================================

def moffat_profile(r_sq, alpha, beta):
    return (1.0 + (r_sq / (alpha**2))) ** (-beta)

def moffat_profile_elliptical(r_sq, alpha, beta):
    return (1.0 + (r_sq / (alpha**2))) ** (-beta)


# =============================================================================
# BASELINE IMAGE BUILDER
# =============================================================================
def build_image_baseline(p):
    margin = 100
    true_x = np.random.uniform(margin, IMG_SIZE - margin)
    true_y = np.random.uniform(margin, IMG_SIZE - margin)

    y, x = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing="ij")
    dx = x - true_x
    dy = y - true_y
    r_raw = np.sqrt(dx**2 + dy**2)
    phi_raw = np.arctan2(dy, dx) 

    theta = p["theta_sun"] 

    sigma_nucleus = p["nucleus_fwhm"] / 2.355
    nucleus_signal = 0.35 * np.exp(-0.5 * (r_raw / sigma_nucleus)**2)

    delta_phi = np.arctan2(np.sin(phi_raw - theta), np.cos(phi_raw - theta))
    jet_width = np.radians(p["jet_opening_angle"])
    jet_modulation = np.exp(-0.5 * (delta_phi / (jet_width / 2.0))**2)
    r_peak = p["jet_peak_distance"]
    jet_profile = (r_raw / r_peak) * np.exp(1.0 - (r_raw / r_peak))
    jet_signal = 0.50 * jet_modulation * jet_profile

    center_offset_x = true_x + p["drift_rate"] * r_raw * np.cos(theta)
    center_offset_y = true_y + p["drift_rate"] * r_raw * np.sin(theta)
    dx_coma = x - center_offset_x
    dy_coma = y - center_offset_y

    u_c = dx_coma * np.cos(theta) + dy_coma * np.sin(theta)
    v_c = -dx_coma * np.sin(theta) + dy_coma * np.cos(theta)
    r_coma_aligned = np.sqrt((u_c / p["coma_aspect_ratio"])**2 + v_c**2)
    coma_signal = 0.30 / np.sqrt((r_coma_aligned / 2.2)**2 + 1.0)

    comet_raw = (nucleus_signal + jet_signal + coma_signal) * np.exp(-r_coma_aligned / p["fade_radius_global"])
    comet_conv = gaussian_filter(comet_raw, sigma=0.75)
    comet_conv = (comet_conv / np.max(comet_conv)) * p["target_peak_above_bg"]

    sky_base = np.full((IMG_SIZE, IMG_SIZE), p["sky_bg_base"], dtype=np.float64)
    vignetting_mode = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
    norm_x = x / IMG_SIZE
    norm_y = y / IMG_SIZE
    
    if vignetting_mode == 0:
        g1, g2 = (1.0 - norm_x) * (1.0 - norm_y), 0.0
    elif vignetting_mode == 1:
        g1, g2 = norm_x * (1.0 - norm_y), 0.0
    elif vignetting_mode == 2:
        g1, g2 = norm_x * norm_y, 0.0
    elif vignetting_mode == 3:
        g1, g2 = (1.0 - norm_x) * norm_y, 0.0
    elif vignetting_mode == 4:
        g1, g2 = (1.0 - norm_x) * (1.0 - norm_y), norm_x * (1.0 - norm_y)
    elif vignetting_mode == 5:
        g1, g2 = norm_x * (1.0 - norm_y), norm_x * norm_y
    elif vignetting_mode == 6:
        g1, g2 = norm_x * norm_y, (1.0 - norm_x) * norm_y
    else:
        g1, g2 = (1.0 - norm_x) * norm_y, (1.0 - norm_x) * (1.0 - norm_y)
        
    sky_base += (g1**1.2) * p["corner_boost1"] + (g2**1.2) * p["corner_boost2"]

    if p["has_dark_blob"]:
        while True:
            bx = np.random.uniform(60, IMG_SIZE - 60)
            by = np.random.uniform(60, IMG_SIZE - 60)
            dist_from_comet = np.sqrt((bx - true_x)**2 + (by - true_y)**2)
            if dist_from_comet >= 75.0:
                break
                
        blob_angle = np.random.uniform(0, np.pi)
        dx_b = x - bx
        dy_b = y - by
        u_b = dx_b * np.cos(blob_angle) + dy_b * np.sin(blob_angle)
        v_b = -dx_b * np.sin(blob_angle) + dy_b * np.cos(blob_angle)
        r_blob_sq = (u_b / p["dark_blob_aspect"])**2 + v_b**2
        blob = p["dark_blob_dip"] * np.exp(-0.5 * (r_blob_sq / (p["dark_blob_sigma"]**2)))
        sky_base += blob

    mosaic_pattern = gaussian_filter(
        np.random.normal(0, 1, size=(IMG_SIZE, IMG_SIZE)), sigma=1.1
    ) * p["sky_mosaic_amplitude"]

    image_base = sky_base + mosaic_pattern + comet_conv

    for star_class in p["star_classes"]:
        for _ in range(star_class["count"]):
            sx = np.random.uniform(15, IMG_SIZE - 15)
            sy = np.random.uniform(15, IMG_SIZE - 15)
            s_amp = np.random.uniform(*star_class["amp"])
            alpha = np.random.uniform(*star_class["alpha_range"])
            beta = np.random.uniform(*star_class["beta_range"])
            
            r_sq = (x - sx)**2 + (y - sy)**2
            image_base += s_amp * moffat_profile(r_sq, alpha=alpha, beta=beta)

    noisy_image = np.random.poisson(
        np.maximum(image_base, 1).astype(np.int64)
    ).astype(np.float64)
    noisy_image += np.random.normal(0, p["noise_std"], size=(IMG_SIZE, IMG_SIZE))

    for _ in range(p["num_hot_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 150000.0
    for _ in range(p["num_dead_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 0.0

    final_image = np.maximum(noisy_image, 0.0).astype(np.float32)

    ix, iy = int(round(true_x)), int(round(true_y))
    r_chk = 25
    y1, y2 = max(0, iy - r_chk), min(IMG_SIZE, iy + r_chk)
    x1, x2 = max(0, ix - r_chk), min(IMG_SIZE, ix + r_chk)
    roi = final_image[y1:y2, x1:x2]
    py_l, px_l = np.unravel_index(np.argmax(roi), roi.shape)
    peak_x, peak_y = x1 + px_l, y1 + py_l

    is_coincident = (int(round(true_x)) == peak_x and int(round(true_y)) == peak_y)

    return final_image, true_x, true_y, is_coincident, p["cat_name"]


# =============================================================================
# LOW SNR IMAGE BUILDER
# =============================================================================
def build_image_low_snr(p):
    margin = 50
    true_x = np.random.uniform(margin, IMG_SIZE - margin)
    true_y = np.random.uniform(margin, IMG_SIZE - margin)

    y, x = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing="ij")
    dx = x - true_x
    dy = y - true_y
    r_raw = np.sqrt(dx**2 + dy**2)

    theta = p["theta_sun"]

    sigma_nucleus = p["nucleus_fwhm"] / 2.355
    nucleus_signal = 0.45 * np.exp(-0.5 * (r_raw / sigma_nucleus)**2)

    center_offset_x = true_x + p["drift_rate"] * r_raw * np.cos(theta)
    center_offset_y = true_y + p["drift_rate"] * r_raw * np.sin(theta)
    dx_c, dy_c = x - center_offset_x, y - center_offset_y
    u_c = dx_c * np.cos(theta) + dy_c * np.sin(theta)
    v_c = -dx_c * np.sin(theta) + dy_c * np.cos(theta)
    r_coma = np.sqrt((u_c / p["coma_aspect_ratio"])**2 + v_c**2)

    coma_signal = 0.55 / np.sqrt((r_coma / 1.8)**2 + 1.0)
    comet_raw = (nucleus_signal + coma_signal) * np.exp(-r_coma / p["fade_radius_global"])
    
    comet_conv = gaussian_filter(comet_raw, sigma=0.55)
    comet_conv = (comet_conv / np.max(comet_conv)) * p["target_peak_above_bg"]

    sky_base = np.full((IMG_SIZE, IMG_SIZE), p["sky_bg_base"], dtype=np.float64)
    mosaic_pattern = gaussian_filter(
        np.random.normal(0, 1, size=(IMG_SIZE, IMG_SIZE)), sigma=1.0
    ) * p["sky_mosaic_amplitude"]

    image_base = sky_base + mosaic_pattern + comet_conv

    for star_class in p["star_classes"]:
        for _ in range(star_class["count"]):
            sx = np.random.uniform(5, IMG_SIZE - 5)
            sy = np.random.uniform(5, IMG_SIZE - 5)
            
            s_amp = np.random.uniform(*star_class["amp"])
            alpha = np.random.uniform(*star_class["alpha_range"])
            beta = np.random.uniform(*star_class["beta_range"])
            
            aspect_ratio = np.random.uniform(1.00, 1.18)
            star_angle = np.random.uniform(0, np.pi)
            
            dx_s = x - sx
            dy_s = y - sy
            u_s = dx_s * np.cos(star_angle) + dy_s * np.sin(star_angle)
            v_s = -dx_s * np.sin(star_angle) + dy_s * np.cos(star_angle)
            r_sq_elliptical = (u_s / aspect_ratio)**2 + v_s**2
            
            image_base += s_amp * moffat_profile_elliptical(r_sq_elliptical, alpha=alpha, beta=beta)

    noisy_image = np.random.poisson(np.maximum(image_base, 1).astype(np.int64)).astype(np.float64)
    noisy_image += np.random.normal(0, p["noise_std"], size=(IMG_SIZE, IMG_SIZE))

    for _ in range(p["num_hot_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 65000.0
    for _ in range(p["num_dead_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 0.0

    final_image = np.maximum(noisy_image, 0.0).astype(np.float32)

    ix, iy = int(round(true_x)), int(round(true_y))
    roi = final_image[max(0, iy - 15):min(IMG_SIZE, iy + 15), max(0, ix - 15):min(IMG_SIZE, ix + 15)]
    py_l, px_l = np.unravel_index(np.argmax(roi), roi.shape)
    peak_x, peak_y = max(0, ix - 15) + px_l, max(0, iy - 15) + py_l

    return final_image, true_x, true_y, (int(round(true_x)) == peak_x and int(round(true_y)) == peak_y), p["cat_name"]


# =============================================================================
# ASYMMETRIC IMAGE BUILDER
# =============================================================================
def build_image_asymmetric(p):
    margin = 80
    true_x = np.random.uniform(margin, IMG_SIZE - margin)
    true_y = np.random.uniform(margin, IMG_SIZE - margin)

    y, x = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing="ij")
    dx = x - true_x
    dy = y - true_y

    theta = p["theta_sun"] 

    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)

    r_core_oval = np.sqrt((u / p["core_oval_ratio"])**2 + v**2)
    sigma_nucleus = p["nucleus_fwhm"] / 2.355
    nucleus_signal = 0.35 * np.exp(-0.5 * (r_core_oval / sigma_nucleus)**2)

    coma_base_signal = 2.85 / np.sqrt((r_core_oval / 2.2)**2 + 1.0)
    sigmoid_stretch = 1.0 + (p["tail_extension_ratio"] - 1.0) / (1.0 + np.exp(-u / p["transition_width"]))
    r_asymmetric = np.sqrt((u / (p["core_oval_ratio"] * sigmoid_stretch))**2 + v**2)

    comet_raw = (nucleus_signal + coma_base_signal) * np.exp(-r_asymmetric / p["fade_radius_base"])
    comet_conv = gaussian_filter(comet_raw, sigma=0.65)
    comet_conv = (comet_conv / np.max(comet_conv)) * p["target_peak_above_bg"]

    sky_base = np.full((IMG_SIZE, IMG_SIZE), p["sky_bg_base"], dtype=np.float64)
    mosaic_pattern = gaussian_filter(
        np.random.normal(0, 1, size=(IMG_SIZE, IMG_SIZE)), sigma=1.0
    ) * p["sky_mosaic_amplitude"]

    image_base = sky_base + mosaic_pattern + comet_conv

    for star_class in p["star_classes"]:
        for _ in range(star_class["count"]):
            sx = np.random.uniform(10, IMG_SIZE - 10)
            sy = np.random.uniform(10, IMG_SIZE - 10)
            
            s_amp = np.random.uniform(*star_class["amp"])
            alpha = np.random.uniform(*star_class["alpha_range"])
            beta = np.random.uniform(*star_class["beta_range"])
            
            aspect_ratio = np.random.uniform(1.00, 1.18)
            star_angle = np.random.uniform(0, np.pi)
            
            dx_s = x - sx
            dy_s = y - sy
            u_s = dx_s * np.cos(star_angle) + dy_s * np.sin(star_angle)
            v_s = -dx_s * np.sin(star_angle) + dy_s * np.cos(star_angle)
            r_sq_elliptical = (u_s / aspect_ratio)**2 + v_s**2
            
            image_base += s_amp * moffat_profile_elliptical(r_sq_elliptical, alpha=alpha, beta=beta)

    noisy_image = np.random.poisson(np.maximum(image_base, 1).astype(np.int64)).astype(np.float64)
    noisy_image += np.random.normal(0, p["noise_std"], size=(IMG_SIZE, IMG_SIZE))

    for _ in range(p["num_hot_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 65000.0
    for _ in range(p["num_dead_pixels"]):
        noisy_image[np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)] = 0.0

    final_image = np.maximum(noisy_image, 0.0).astype(np.float32)

    ix, iy = int(round(true_x)), int(round(true_y))
    roi = final_image[max(0, iy - 20):min(IMG_SIZE, iy + 20), max(0, ix - 20):min(IMG_SIZE, ix + 20)]
    py_l, px_l = np.unravel_index(np.argmax(roi), roi.shape)
    peak_x, peak_y = max(0, ix - 20) + px_l, max(0, iy - 20) + py_l

    return final_image, true_x, true_y, (int(round(true_x)) == peak_x and int(round(true_y)) == peak_y), p["cat_name"]


# =============================================================================
# SYNTHETIC IMAGE ROUTER
# =============================================================================
def create_synthetic_image(comet_type):
    if comet_type == 0:
        params = get_params_baseline()
        return build_image_baseline(params)
    elif comet_type == 1:
        params = get_params_low_snr()
        return build_image_low_snr(params)
    else:
        params = get_params_asymmetric()
        return build_image_asymmetric(params)


# =============================================================================
# MAIN EXECUTION ROUTINE
# =============================================================================
def main():
    np.random.seed(42)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    truth_records = []
    coincidents = 0

    per_category = NUM_IMAGES // 3
    print(f"Generating {NUM_IMAGES} synthetic scientific images...")
    print(f"Distribution: {per_category} per category (Baseline, Low_SNR, Asymmetric)")
    print("=" * 65)

    start_time = datetime(2026, 3, 27, 22, 0, 0)

    for i in range(NUM_IMAGES):
        if i < per_category:
            comet_type = 0
            idx_in_cat = i
        elif i < 2 * per_category:
            comet_type = 1
            idx_in_cat = i - per_category
        else:
            comet_type = 2
            idx_in_cat = i - (2 * per_category)

        img_data, tx, ty, is_coincident, cat_name = create_synthetic_image(comet_type)
        
        filename = f"synth_{idx_in_cat:03d}_{cat_name}.fits"

        if is_coincident:
            coincidents += 1

        frame_time = start_time + timedelta(minutes=(i * 3))
        date_obs_str = frame_time.strftime("%Y-%m-%dT%H:%M:%S.000")

        if cat_name == "Baseline":
            hist_msg = f"Type: {cat_name} (Coherent Physical Jet Emission & Shifted Peak)"
        elif cat_name == "Low_SNR":
            hist_msg = f"Type: {cat_name} (Variable Sky BG & Sub-Pixel Nucleus Drift)"
        else:
            hist_msg = f"Type: {cat_name} (High Visibility Sigmoid Tail Extension Asymmetric Model)"

        header = fits.Header(
            {
                "BITPIX": -32,
                "DATE-OBS": date_obs_str,
                "HISTORY": hist_msg,
            }
        )
        fits.PrimaryHDU(img_data, header=header).writeto(
            os.path.join(OUTPUT_DIR, filename), overwrite=True
        )

        truth_records.append([filename, tx, ty, cat_name])

        if (i + 1) % 50 == 0 or (i + 1) == NUM_IMAGES:
            print(f"[*] Generated {i + 1}/{NUM_IMAGES} images...")

    csv_path = TRUTH_FILE
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_x", "true_y", "category"])
        writer.writerows(truth_records)

    print("=" * 65)
    print("SCIENTIFIC DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print(f"[+] Images saved to:        {OUTPUT_DIR}")
    print(f"[+] Ground truth saved to: {csv_path}")
    print(f"[i] Coincident peaks with true center: {coincidents} out of {NUM_IMAGES}")
    print("=" * 65)


if __name__ == "__main__":
    main()