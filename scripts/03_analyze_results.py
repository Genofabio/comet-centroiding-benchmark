import csv
import os
import sys
import numpy as np


# =============================================================================
# BENCHMARK REANALYZER CLASS
# =============================================================================
class BenchmarkReanalyzer:
    def __init__(
        self,
        raw_csv="results/results_benchmark_raw_data.csv",
        output_txt="results/results_benchmark_rankings.txt",
    ):
        self.raw_csv = raw_csv
        self.output_txt = output_txt
        
        os.makedirs(os.path.dirname(self.raw_csv) if os.path.dirname(self.raw_csv) else "results", exist_ok=True)
        
        self.raw_data = []
        self.algorithms = set()
        self.log_lines = []

        self.weights = {
            "click_drift": 0.3,         
            "best_accuracy": 0.3,       
            "window_sensitivity": 0.2,  
            "morpho_insensitivity": 0.15, 
            "execution_speed": 0.05,    
        }

    def _log_to_file(self, message=""):
        """Saves message to internal log buffer for txt generation without console output."""
        self.log_lines.append(message)

    def load_raw_data(self):
        if not os.path.exists(self.raw_csv):
            print(f"[ERROR] Data file not found: {self.raw_csv}")
            sys.exit(1)

        with open(self.raw_csv, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["error_px"] == "FAILED" or row["error_px"] == "inf":
                    continue
                row["error_px"] = float(row["error_px"])
                row["time_ms"] = float(row["time_ms"])
                row["roi_radius"] = int(row["roi_radius"])
                row["offset_dist"] = float(row["offset_dist"])
                self.raw_data.append(row)
                self.algorithms.add(row["algo_name"])

        self.algorithms = sorted(list(self.algorithms))
        print(f"[INFO] Loaded {len(self.raw_data)} valid records for {len(self.algorithms)} algorithms.")

    def compute_orthogonal_metrics(self):
        raw_metrics = {}

        for algo in self.algorithms:
            algo_rows = [r for r in self.raw_data if r["algo_name"] == algo]

            w_rows = [r for r in algo_rows if r["test_type"] == "WINDOW"]
            w_5 = [r["error_px"] for r in w_rows if r["roi_radius"] == 5]
            w_10 = [r["error_px"] for r in w_rows if r["roi_radius"] == 10]
            w_20 = [r["error_px"] for r in w_rows if r["roi_radius"] == 20]
            w_30 = [r["error_px"] for r in w_rows if r["roi_radius"] == 30]

            o_rows = [r for r in algo_rows if r["test_type"] == "OFFSET"]
            o_00 = [r["error_px"] for r in o_rows if r["offset_dist"] == 0.0]
            o_05 = [r["error_px"] for r in o_rows if r["offset_dist"] == 0.5]
            o_10 = [r["error_px"] for r in o_rows if r["offset_dist"] == 1.0]
            o_15 = [r["error_px"] for r in o_rows if r["offset_dist"] == 1.5]

            cat_baseline = [r["error_px"] for r in o_rows if r["category"] == "Baseline" and r["offset_dist"] == 0.0]
            cat_lowsnr = [r["error_px"] for r in o_rows if r["category"] == "Low_SNR" and r["offset_dist"] == 0.0]
            cat_asymm = [r["error_px"] for r in o_rows if r["category"] == "Asymmetric" and r["offset_dist"] == 0.0]

            times = [r["time_ms"] for r in algo_rows]

            med_offsets = [np.median(o_00), np.median(o_05), np.median(o_10), np.median(o_15)]
            click_drift_std = np.std(med_offsets)

            med_w5 = np.median(w_5) if w_5 else float('inf')
            med_w10 = np.median(w_10) if w_10 else float('inf')
            best_accuracy_med = min(med_w5, med_w10)
            best_window_label = "5 px" if med_w5 <= med_w10 else "10 px"

            med_windows = [med_w5, med_w10, np.median(w_20), np.median(w_30)]
            window_var_range = max(med_windows) - min(med_windows)

            med_categories = [np.median(cat_baseline), np.median(cat_lowsnr), np.median(cat_asymm)]
            category_var_range = max(med_categories) - min(med_categories)

            mean_time_ms = np.mean(times)

            raw_metrics[algo] = {
                "med_w5": med_w5,
                "med_w10": med_w10,
                "med_w20": np.median(w_20),
                "med_w30": np.median(w_30),
                "med_o00": np.median(o_00),
                "med_o05": np.median(o_05),
                "med_o10": np.median(o_10),
                "med_o15": np.median(o_15),
                "med_baseline": np.median(cat_baseline),
                "med_lowsnr": np.median(cat_lowsnr),
                "med_asymm": np.median(cat_asymm),
                "raw_click_drift": click_drift_std,
                "raw_best_acc": best_accuracy_med,
                "best_window": best_window_label,
                "raw_window_var": window_var_range,
                "raw_cat_var": category_var_range,
                "raw_time_ms": mean_time_ms,
            }

        def norm_100(raw_dict, key):
            vals = [m[key] for m in raw_dict.values() if m[key] != float('inf')]
            if not vals: return {algo: 0.0 for algo in raw_dict}
            min_v, max_v = min(vals), max(vals)
            rng = max_v - min_v if max_v != min_v else 1.0
            return {algo: 100.0 * (1.0 - (m[key] - min_v) / rng) if m[key] != float('inf') else 0.0 for algo, m in raw_dict.items()}

        s_drift = norm_100(raw_metrics, "raw_click_drift")
        s_acc = norm_100(raw_metrics, "raw_best_acc")
        s_winvar = norm_100(raw_metrics, "raw_window_var")
        s_catvar = norm_100(raw_metrics, "raw_cat_var")
        s_time = norm_100(raw_metrics, "raw_time_ms")

        metrics = {}
        for algo, m in raw_metrics.items():
            composite = (
                (self.weights["click_drift"] * s_drift[algo])
                + (self.weights["best_accuracy"] * s_acc[algo])
                + (self.weights["window_sensitivity"] * s_winvar[algo])
                + (self.weights["morpho_insensitivity"] * s_catvar[algo])
                + (self.weights["execution_speed"] * s_time[algo])
            )

            metrics[algo] = m
            metrics[algo]["s_drift"] = s_drift[algo]
            metrics[algo]["s_acc"] = s_acc[algo]
            metrics[algo]["s_winvar"] = s_winvar[algo]
            metrics[algo]["s_catvar"] = s_catvar[algo]
            metrics[algo]["s_time"] = s_time[algo]
            metrics[algo]["composite_score"] = composite

        return metrics

    def generate_report(self):
        self.load_raw_data()
        metrics = self.compute_orthogonal_metrics()

        self._log_to_file("=" * 150)
        self._log_to_file("    COMETARY CENTROIDING ALGORITHMS BENCHMARK ANALYSIS (DIMENSIONLESS ORTHOGONAL MATRIX 0-100)")
        self._log_to_file("=" * 150)

        # SECTION 1: INITIAL CLICK SHIFT INVARIANCE
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 1. INITIAL SHIFT / CLICK INVARIANCE (Stability at the trigger point)")
        self._log_to_file("-" * 150)
        self._log_to_file(
            f" {'ALGORITHM':<45} | {'0.00 px':<8} | {'0.50 px':<8} | {'1.00 px':<8} | {'1.50 px':<8} | {'DRIFT (Std)':<12} | {'DRIFT SCORE (30%)':<18}"
        )
        self._log_to_file("-" * 150)

        sorted_drift = sorted(self.algorithms, key=lambda a: metrics[a]["s_drift"], reverse=True)
        for algo in sorted_drift:
            m = metrics[algo]
            self._log_to_file(
                f" {algo:<45} | {m['med_o00']:.3f} px  | {m['med_o05']:.3f} px  | {m['med_o10']:.3f} px  | {m['med_o15']:.3f} px  | {m['raw_click_drift']:.5f} px  | {m['s_drift']:6.2f} / 100"
            )

        # SECTION 2: OPTIMAL ACCURACY ACROSS DATASET
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 2. OPTIMAL NOMINAL ACCURACY (Evaluated on the whole dataset, no offset, best ROI between 5px and 10px)")
        self._log_to_file("-" * 150)
        self._log_to_file(
            f" {'ALGORITHM':<45} | {'ERROR (5px)':<14} | {'ERROR (10px)':<15} | {'BEST ROI':<15} | {'ACCURACY SCORE (30%)':<25}"
        )
        self._log_to_file("-" * 150)

        sorted_acc = sorted(self.algorithms, key=lambda a: metrics[a]["s_acc"], reverse=True)
        for algo in sorted_acc:
            m = metrics[algo]
            self._log_to_file(
                f" {algo:<45} | {m['med_w5']:.4f} px     | {m['med_w10']:.4f} px      | {m['best_window']:<15} | {m['s_acc']:6.2f} / 100"
            )

        # SECTION 3: WINDOW SIZE SENSITIVITY
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 3. WINDOW SIZE INVARIANCE (Stability across ROIs of 5, 10, 20, 30 px)")
        self._log_to_file("-" * 150)
        self._log_to_file(
            f" {'ALGORITHM':<45} | {'5 px':<8} | {'10 px':<8} | {'20 px':<8} | {'30 px':<8} | {'RANGE (Max-Min)':<15} | {'WIN-VAR SCORE (20%)':<19}"
        )
        self._log_to_file("-" * 150)

        sorted_winvar = sorted(self.algorithms, key=lambda a: metrics[a]["s_winvar"], reverse=True)
        for algo in sorted_winvar:
            m = metrics[algo]
            self._log_to_file(
                f" {algo:<45} | {m['med_w5']:.3f} pt| {m['med_w10']:.3f} pt| {m['med_w20']:.3f} pt| {m['med_w30']:.3f} pt| {m['raw_window_var']:.4f} px      | {m['s_winvar']:6.2f} / 100"
            )

        # SECTION 4: HOMOGENEITY ACROSS COMET CLASSES
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 4. MORPHOLOGICAL INVARIANCE (Performance discrepancy across Baseline, Low_SNR, Asymmetric classes)")
        self._log_to_file("-" * 150)
        self._log_to_file(
            f" {'ALGORITHM':<45} | {'BASELINE':<10} | {'LOW_SNR':<10} | {'ASYMM':<9} | {'DISCREPANCY (Range)':<22} | {'MORPHO SCORE (15%)':<18}"
        )
        self._log_to_file("-" * 150)

        sorted_catvar = sorted(self.algorithms, key=lambda a: metrics[a]["s_catvar"], reverse=True)
        for algo in sorted_catvar:
            m = metrics[algo]
            self._log_to_file(
                f" {algo:<45} | {m['med_baseline']:.3f} px | {m['med_lowsnr']:.3f} px | {m['med_asymm']:.3f} px | {m['raw_cat_var']:.4f} px               | {m['s_catvar']:6.2f} / 100"
            )

        # SECTION 5: FINAL GLOBAL RANKING
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 5. FINAL WEIGHTED RANKING (DIMENSIONLESS MULTI-CRITERIA MATRIX 0-100)")
        self._log_to_file(
            f"    [Weights: Click Invar. = {self.weights['click_drift']*100:.0f}% | Optimal Acc. = {self.weights['best_accuracy']*100:.0f}% | "
            f"ROI Invar. = {self.weights['window_sensitivity']*100:.0f}% | Morpho Invar. = {self.weights['morpho_insensitivity']*100:.0f}% | "
            f"Speed = {self.weights['execution_speed']*100:.0f}%]"
        )
        self._log_to_file("=" * 150)

        sorted_final = sorted(self.algorithms, key=lambda a: metrics[a]["composite_score"], reverse=True)

        for rank, algo in enumerate(sorted_final, start=1):
            m = metrics[algo]
            self._log_to_file(
                f" {rank:2d}. {algo:<45} | SCORE: {m['composite_score']:6.2f} / 100 | "
                f"[ClickDrift: {m['s_drift']:5.1f}pt | OptAcc: {m['s_acc']:5.1f}pt | "
                f"InvarROI: {m['s_winvar']:5.1f}pt | InvarMorph: {m['s_catvar']:5.1f}pt | "
                f"Speed: {m['s_time']:5.1f}pt ({m['raw_time_ms']:.1f}ms)]"
            )

        # SECTION 6: CRITICAL SYNTHESIS
        self._log_to_file("\n" + "-" * 150)
        self._log_to_file(" 6. CRITICAL SYNTHESIS OF RESULTS")
        self._log_to_file("=" * 150)

        top1 = sorted_final[0]
        top2 = sorted_final[1]

        classic_keywords = ["Barycenter", "Interpolation", "Gradients", "Mapping", "Decomposition"]
        classic_algos = [a for a in self.algorithms if any(k in a for k in classic_keywords)]

        self._log_to_file(f" [*] BEST OVERALL ALGORITHM:           '{top1}' (Composite Score: {metrics[top1]['composite_score']:.2f} / 100)")
        self._log_to_file(f" [*] RUNNER-UP:                       '{top2}' (Composite Score: {metrics[top2]['composite_score']:.2f} / 100)")

        if classic_algos:
            top_classic = max(classic_algos, key=lambda a: metrics[a]["composite_score"])
            self._log_to_file(f" [*] BEST EMPIRICAL ALGORITHM (NO FIT): '{top_classic}' (Composite Score: {metrics[top_classic]['composite_score']:.2f} / 100)")

        self._log_to_file("=" * 150 + "\n")

        with open(self.output_txt, "w") as f:
            f.write("\n".join(self.log_lines))

        print(f"[INFO] Reanalysis completed successfully. Updated report saved to: {self.output_txt}")


if __name__ == "__main__":
    analyzer = BenchmarkReanalyzer()
    analyzer.generate_report()