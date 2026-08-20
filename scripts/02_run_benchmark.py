import os
import csv
import time
import numpy as np

from src.io_utils import load_fits_image
from src.empirical_methods import (
    ClassicAlgorithmType,
    find_center_global_luminous_barycenter,
    find_center_adaptive_statistical_threshold_barycenter,
    find_center_local_parabolic_interpolation,
    find_center_intersection_of_radial_gradients,
    find_center_moving_window_symmetry_mapping,
    find_center_nucleus_coma_decomposition
)
from src.parametric_methods import (
    AlgorithmType,
    find_center_two_dimensional_gaussian_fitting, 
    find_center_one_dimensional_power_law_profile,
    find_center_truncated_radial_power_law_profile,
    find_center_central_softened_radial_profile,
    find_center_softened_and_rotatable_elliptical_profile,
    find_center_asymmetric_rotational_hybrid_profile,
    find_center_bounded_softened_radial_profile,
    find_center_anisotropic_orthogonal_profile,
    find_center_asymmetric_quadrant_profile,
    find_center_asymmetric_weighted_residual_profile
)


# =============================================================================
# ALGORITHMS DICTIONARY
# =============================================================================
ALL_ALGORITHMS = {
    ClassicAlgorithmType.GLOBAL_LUMINOUS_BARYCENTER: find_center_global_luminous_barycenter,
    ClassicAlgorithmType.ADAPTIVE_STATISTICAL_THRESHOLD_BARYCENTER: find_center_adaptive_statistical_threshold_barycenter,
    ClassicAlgorithmType.LOCAL_PARABOLIC_INTERPOLATION: find_center_local_parabolic_interpolation,
    ClassicAlgorithmType.INTERSECTION_OF_RADIAL_GRADIENTS: find_center_intersection_of_radial_gradients,
    ClassicAlgorithmType.MOVING_WINDOW_SYMMETRY_MAPPING: find_center_moving_window_symmetry_mapping,
    ClassicAlgorithmType.NUCLEUS_COMA_DECOMPOSITION: find_center_nucleus_coma_decomposition,
    
    AlgorithmType.TWO_DIMENSIONAL_GAUSSIAN_FITTING: find_center_two_dimensional_gaussian_fitting,
    AlgorithmType.ONE_DIMENSIONAL_POWER_LAW_PROFILE: find_center_one_dimensional_power_law_profile,
    AlgorithmType.TRUNCATED_RADIAL_POWER_LAW_PROFILE: find_center_truncated_radial_power_law_profile,
    AlgorithmType.CENTRAL_SOFTENED_RADIAL_PROFILE: find_center_central_softened_radial_profile,
    AlgorithmType.SOFTENED_AND_ROTATABLE_ELLIPTICAL_PROFILE: find_center_softened_and_rotatable_elliptical_profile,
    AlgorithmType.ASYMMETRIC_ROTATIONAL_HYBRID_PROFILE: find_center_asymmetric_rotational_hybrid_profile,
    AlgorithmType.BOUNDED_SOFTENED_RADIAL_PROFILE: find_center_bounded_softened_radial_profile,
    AlgorithmType.ANISOTROPIC_ORTHOGONAL_PROFILE: find_center_anisotropic_orthogonal_profile,
    AlgorithmType.ASYMMETRIC_QUADRANT_PROFILE: find_center_asymmetric_quadrant_profile,
    AlgorithmType.ASYMMETRIC_WEIGHTED_RESIDUAL_PROFILE: find_center_asymmetric_weighted_residual_profile
}


# =============================================================================
# AUTOMATED BENCHMARK CLASS
# =============================================================================
class CometAlgorithmTester:
    def __init__(self, input_dir="data/synthetic_test", truth_file="data/synthetic_ground_truth.csv"):
        self.input_dir = input_dir
        self.truth_file = truth_file
        
        os.makedirs("results", exist_ok=True)
        
        self.output_csv = os.path.join("results", "results_benchmark_raw_data.csv")
        self.output_txt = os.path.join("results", "results_benchmark_rankings.txt")
        
        self.algorithms = ALL_ALGORITHMS
        
        self.zoom_radii = [5, 10, 20, 30]
        self.fixed_radius = 10  
        self.offset_distances = [0.0, 0.5, 1.0, 1.5]
        
        self.ground_truth = {}
        self.all_results = []  
        self.log_lines = [] 

    def _log_to_file(self, message):
        """Saves message to internal log buffer for txt generation without console output."""
        self.log_lines.append(message)

    def load_ground_truth(self):
        if not os.path.exists(self.truth_file):
            raise FileNotFoundError(f"Ground Truth file not found: {self.truth_file}")
            
        with open(self.truth_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.ground_truth[row['filename']] = {
                    'x': float(row['true_x']),
                    'y': float(row['true_y']),
                    'category': row['category']
                }
        print(f"[INFO] Loaded Ground Truth for {len(self.ground_truth)} images.")

    def find_subpixel_peak(self, img_data, true_x, true_y, search_radius=3):
        h, w = img_data.shape
        ix, iy = int(round(true_x)), int(round(true_y))
        
        y1, y2 = max(0, iy - search_radius), min(h, iy + search_radius)
        x1, x2 = max(0, ix - search_radius), min(w, ix + search_radius)
        roi = img_data[y1:y2, x1:x2]
        
        py_local, px_local = np.unravel_index(np.argmax(roi), roi.shape)
        peak_x_int = x1 + px_local
        peak_y_int = y1 + py_local
        
        y1_t, y2_t = max(0, peak_y_int - 1), min(h, peak_y_int + 2)
        x1_t, x2_t = max(0, peak_x_int - 1), min(w, peak_x_int + 2)
        tiny_roi = img_data[y1_t:y2_t, x1_t:x2_t]
        
        y_indices, x_indices = np.indices(tiny_roi.shape)
        flux = np.sum(tiny_roi)
        if flux == 0: flux = 1
        
        cx = np.sum(x_indices * tiny_roi) / flux
        cy = np.sum(y_indices * tiny_roi) / flux
        
        return x1_t + cx, y1_t + cy

    def run_algorithm(self, func, algo_enum, roi_data, x1, y1, test_type, roi_radius, offset_dist, fname, ref_x, ref_y):
        start_time = time.perf_counter()
        try:
            res = func(roi_data, x1, y1)
            x_f, y_f = res[0], res[1]
        except Exception:
            x_f, y_f = float('inf'), float('inf')
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        true_x = self.ground_truth[fname]['x']
        true_y = self.ground_truth[fname]['y']
        cat = self.ground_truth[fname]['category']
        
        error_px = np.sqrt((x_f - true_x)**2 + (y_f - true_y)**2) if x_f != float('inf') else float('inf')
        algo_name = algo_enum.value if hasattr(algo_enum, 'value') else str(algo_enum)

        self.all_results.append({
            'filename': fname,
            'category': cat,
            'algo_name': algo_name,
            'algo_enum': algo_enum,
            'test_type': test_type,
            'roi_radius': roi_radius,
            'offset_dist': round(offset_dist, 4),
            'ref_x': round(ref_x, 4),
            'ref_y': round(ref_y, 4),
            'true_x': round(true_x, 4),
            'true_y': round(true_y, 4),
            'found_x': round(x_f, 4) if x_f != float('inf') else 'FAILED',
            'found_y': round(y_f, 4) if y_f != float('inf') else 'FAILED',
            'error_px': round(error_px, 4) if error_px != float('inf') else 'FAILED',
            'time_ms': round(elapsed_ms, 2)
        })

    def generate_random_offsets(self, peak_x, peak_y):
        generated = []
        for dist in self.offset_distances:
            if dist == 0.0:
                generated.append((peak_x, peak_y, 0.0))
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)
                generated.append((peak_x + dx, peak_y + dy, dist))
        return generated

    def run_tests(self):
        self.load_ground_truth()
        filenames = sorted(list(self.ground_truth.keys()))

        if not filenames: 
            print("[WARNING] No images found.")
            return

        total_files = len(filenames)
        print(f"\n[INFO] Starting benchmark execution on {total_files} images...")
        print("=" * 65)
        
        for idx, fname in enumerate(filenames, start=1):
            fpath = os.path.join(self.input_dir, fname)
            if not os.path.exists(fpath): 
                continue
                
            img_data = load_fits_image(fpath)
            true_x = self.ground_truth[fname]['x']
            true_y = self.ground_truth[fname]['y']
            h, w = img_data.shape
            
            peak_x, peak_y = self.find_subpixel_peak(img_data, true_x, true_y)

            for r in self.zoom_radii:
                x1, x2 = max(0, int(peak_x - r)), min(w, int(peak_x + r))
                y1, y2 = max(0, int(peak_y - r)), min(h, int(peak_y + r))
                if x2 <= x1 or y2 <= y1: continue
                roi_data = img_data[y1:y2, x1:x2]
                
                for algo_enum, func in self.algorithms.items():
                    self.run_algorithm(func, algo_enum, roi_data, x1, y1, "WINDOW", r, 0.0, fname, peak_x, peak_y)

            r = self.fixed_radius
            image_offsets = self.generate_random_offsets(peak_x, peak_y)
            
            for ref_x, ref_y, offset_dist in image_offsets:
                x1, x2 = max(0, int(ref_x - r)), min(w, int(ref_x + r))
                y1, y2 = max(0, int(ref_y - r)), min(h, int(ref_y + r))
                if x2 <= x1 or y2 <= y1: continue
                roi_data = img_data[y1:y2, x1:x2]
                
                for algo_enum, func in self.algorithms.items():
                    self.run_algorithm(func, algo_enum, roi_data, x1, y1, "OFFSET", r, offset_dist, fname, ref_x, ref_y)

            print(f"[*] Processed image {idx}/{total_files}: {fname}")

        print("=" * 65)
        print("[INFO] Image processing completed successfully.")
        self.save_raw_results()
        self.compute_and_save_rankings()

    def save_raw_results(self):
        if not self.all_results: return
        keys = self.all_results[0].keys()
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.all_results)
        print(f"[INFO] Detailed raw data saved to {self.output_csv}")

    def safe_stats(self, val_list):
        if not val_list:
            return float('inf'), float('inf')
        return np.mean(val_list), np.median(val_list)

    def calc_metrics(self):
        metrics = {algo: {
            'window_errors': {5: [], 10: [], 20: [], 30: []}, 
            'offset_errors': {0.0: [], 0.5: [], 1.0: [], 1.5: []}, 
            'cat_errors': {'Baseline': [], 'Low_SNR': [], 'Asymmetric': []}
        } for algo in self.algorithms}
        
        for r in self.all_results:
            if r['error_px'] == 'FAILED': continue
            algo = r['algo_enum']
            err = float(r['error_px'])
            
            if r['test_type'] == "WINDOW":
                radius = int(r['roi_radius'])
                if radius in metrics[algo]['window_errors']:
                    metrics[algo]['window_errors'][radius].append(err)
                    
            elif r['test_type'] == "OFFSET":
                od = round(float(r['offset_dist']), 1)
                if od in metrics[algo]['offset_errors']:
                    metrics[algo]['offset_errors'][od].append(err)
                
                if r['category'] in metrics[algo]['cat_errors']:
                    metrics[algo]['cat_errors'][r['category']].append(err)

        final_ranking = []
        for algo in self.algorithms:
            algo_data = metrics[algo]
            
            w_stats = {r: self.safe_stats(errs) for r, errs in algo_data['window_errors'].items()}
            all_w_errs = [e for errs in algo_data['window_errors'].values() for e in errs]
            global_w_mean, global_w_med = self.safe_stats(all_w_errs)
            
            o_stats = {od: self.safe_stats(errs) for od, errs in algo_data['offset_errors'].items()}
            all_o_errs = [e for errs in algo_data['offset_errors'].values() for e in errs]
            global_o_mean, global_o_med = self.safe_stats(all_o_errs)
            
            c_stats = {cat: self.safe_stats(errs) for cat, errs in algo_data['cat_errors'].items()}
            
            algo_name = algo.value if hasattr(algo, 'value') else str(algo)
            algo_type = "Classic" if "Classic" in str(type(algo)) else "Fit Model"
            
            final_score = (global_w_med + global_o_med) / 2.0 if (global_w_med != float('inf') and global_o_med != float('inf')) else float('inf')
            
            final_ranking.append({
                'name': algo_name,
                'category': algo_type,
                'w_stats': w_stats,
                'global_w': (global_w_mean, global_w_med),
                'o_stats': o_stats,
                'global_o': (global_o_mean, global_o_med),
                'c_stats': c_stats,
                'final_score': final_score
            })
            
        return final_ranking

    def compute_and_save_rankings(self):
        global_ranking = self.calc_metrics()
        self.log_lines = [] 

        def fmt_med(stats_tuple): 
            return f"{stats_tuple[1]:.3f}" if stats_tuple[1] != float('inf') else "N/A"
        
        def fmt_mm(mean, med): 
            return f"{mean:.2f} / {med:.2f}" if med != float('inf') else "FAILED"

        self._log_to_file("\n" + "#" * 130)
        self._log_to_file(" AUTOMATED ALGORITHM BENCHMARK RESULTS (Using MEDIAN for Robustness)")
        self._log_to_file("#" * 130)

        # 1. WINDOW TEST
        self._log_to_file("\n" + "=" * 130)
        self._log_to_file(" 1. WINDOW SIZE ROBUSTNESS (Median Error in px by ROI Radius)")
        self._log_to_file("=" * 130)
        self._log_to_file(f" {'ALGORITHM':<45} | {'5 px':<10} | {'10 px':<10} | {'20 px':<10} | {'30 px':<10} | {'OVERALL (Mean/Med)':<20}")
        self._log_to_file("-" * 130)
        sorted_window = sorted(global_ranking, key=lambda x: x['global_w'][1]) 
        
        for r in sorted_window:
            w5 = fmt_med(r['w_stats'][5])
            w10 = fmt_med(r['w_stats'][10])
            w20 = fmt_med(r['w_stats'][20])
            w30 = fmt_med(r['w_stats'][30])
            overall = fmt_mm(r['global_w'][0], r['global_w'][1])
            self._log_to_file(f" {r['name']:<45} | {w5:<10} | {w10:<10} | {w20:<10} | {w30:<10} | {overall:<20}")

        # 2. OFFSET TEST
        self._log_to_file("\n" + "=" * 130)
        self._log_to_file(f" 2. OFFSET / CLICK STABILITY (Median Error in px by Initial Shift - Random angle, Fixed {self.fixed_radius}px Radius)")
        self._log_to_file("=" * 130)
        keys = self.offset_distances
        col_headers = [f"{k:.2f} px" for k in keys]
        
        self._log_to_file(f" {'ALGORITHM':<45} | {col_headers[0]:<10} | {col_headers[1]:<10} | {col_headers[2]:<10} | {col_headers[3]:<10} | {'OVERALL (Mean/Med)':<20}")
        self._log_to_file("-" * 130)
        sorted_off = sorted(global_ranking, key=lambda x: x['global_o'][1])
        
        for r in sorted_off:
            o0 = fmt_med(r['o_stats'].get(keys[0], (float('inf'), float('inf'))))
            o1 = fmt_med(r['o_stats'].get(keys[1], (float('inf'), float('inf'))))
            o2 = fmt_med(r['o_stats'].get(keys[2], (float('inf'), float('inf'))))
            o3 = fmt_med(r['o_stats'].get(keys[3], (float('inf'), float('inf'))))
            overall = fmt_mm(r['global_o'][0], r['global_o'][1])
            self._log_to_file(f" {r['name']:<45} | {o0:<10} | {o1:<10} | {o2:<10} | {o3:<10} | {overall:<20}")

        # 3. CATEGORY BREAKDOWN
        self._log_to_file("\n" + "=" * 130)
        self._log_to_file(f" 3. PERFORMANCE BREAKDOWN BY CATEGORY (Mean / Median Error in px - ONLY {self.fixed_radius}px optimal ROI evaluated)")
        self._log_to_file("=" * 130)
        self._log_to_file(f" {'ALGORITHM':<45} | {'BASELINE':<20} | {'LOW_SNR':<20} | {'ASYMMETRIC':<20}")
        self._log_to_file("-" * 130)
        for r in sorted_off:
            c_baseline = fmt_mm(*r['c_stats']['Baseline'])
            c_low_snr = fmt_mm(*r['c_stats']['Low_SNR'])
            c_asymm = fmt_mm(*r['c_stats']['Asymmetric'])
            self._log_to_file(f" {r['name']:<45} | {c_baseline:<20} | {c_low_snr:<20} | {c_asymm:<20}")

        # 4. FINAL GLOBAL RANKING
        self._log_to_file("\n" + "=" * 130)
        self._log_to_file(" 4. ABSOLUTE GLOBAL RANKING (Sorted by Median Error across all tests)")
        self._log_to_file("=" * 130)
        sorted_global = sorted(global_ranking, key=lambda x: x['final_score'])
        for i, r in enumerate(sorted_global):
            if r['final_score'] != float('inf'):
                self._log_to_file(f" {i+1:2d}. {r['name']:<45} [{r['category']:<11}] | Final Score (Median): {r['final_score']:>6.3f} px")
            else:
                self._log_to_file(f" {i+1:2d}. {r['name']:<45} [{r['category']:<11}] | TOTALLY FAILED")
        self._log_to_file("#" * 130 + "\n")

        with open(self.output_txt, "w") as f:
            f.write("\n".join(self.log_lines))
            
        print(f"[INFO] Formatted rankings successfully saved to {self.output_txt}")


if __name__ == "__main__":
    tester = CometAlgorithmTester()
    tester.run_tests()