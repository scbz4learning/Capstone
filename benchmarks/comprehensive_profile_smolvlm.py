import torch
import time
import json
import os
import sys
import threading
import subprocess
import numpy as np
import pandas as pd
import psutil
import argparse
import glob
import datetime
import gc
import csv
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TextIteratorStreamer
from torchvision.transforms import ToPILImage

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# --- Environment check ---
def check_environment(config, args):
    current_os = 'windows' if os.name == 'nt' else 'linux'
    config_os = config.get("os_type", "").lower()
    
    # WSL Orchestration
    if config_os == 'wsl':
        if current_os == 'windows':
            run_wsl_host(config, args)
            sys.exit(0)
        elif not getattr(args, 'wsl_worker', False):
            sys.exit(f"""[Error] Config os_type is '{config_os}', but running on '{current_os}' directly without --wsl-worker flag. 
        If you want to run via WSL, please run this script from Windows so it can orchestrate the WSL environment and record host power!
        Exiting.""")
        return
        
    print(f"[Info] Detected OS: {current_os.capitalize()}")
    if config_os != current_os:
        sys.exit(f"[Error] Config asks for '{config_os}' but running on '{current_os}'. Exiting.")
        
    if current_os == 'linux':
        if os.geteuid() != 0:
            print("[Error] This script requires root privileges to access RAPL power counters on Linux.")
            print("        Please run with: sudo python3 ...")
            sys.exit(1)

# --- Configuration ---
DEFAULT_CONFIG = {
    "os_type": "linux", 
    "task_type": "vision_autoregressive",
    "backend": "pytorch",
    "model": "HuggingFaceTB/SmolVLM-Instruct",
    "output_dir": "./profiling_logs",
    "wsl": {
        "distro": "", 
        "python_path": "python3"
    },
    "execution": {
        "device": "cuda",
        "dtype": ["float16", "bfloat16"],
        "gpu_options": {
            "attn_implementation": ["eager", "sdpa"]
        },
        "cpu_options": {
            "cpu_acc": ["none"]
        },
        "is_integrated": True,
        "sampling_randomness": False,
        "temperature": None,
        "passes": {
            "warmup": {
                "num_warmup": 10
            },
            "end_to_end": {
                "num_test": 20
            },
            "power": {
                "num_test": 10
            }
        }
    },
    "inputs": {
        "prompt": "Describe the images briefly.",
        "prompt_size": 0,
        "images": [],
        "image_size": 384,
        "num_images": 2
    },
    "output": {
        "output_tokens": 128
    },
    "target_metrics": {
        "warmup": ["warmup_latency"],
        "end_to_end": ["TTFT", "TPOT", "Total_latency", "tokens/sec", "peak_mem"],
        "power": ["avg_power", "energy_per_inference", "fps_watt"],
        "bandwidth_analysis": True
    }
}

def dict_deep_update(d, u):
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_timestamped_filename(output_dir, name, ext):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{name}_{ts}.{ext}")

def calculate_metrics(data):
    if not data: return {}
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "p50": np.percentile(data, 50),
        "p95": np.percentile(data, 95),
        "p99": np.percentile(data, 99),
        "std": np.std(data)
    }

def sanity_check_timing(phase_name, timestamps, iteration_counts, result_dict):
    if not timestamps or len(timestamps) < 2:
        return {"error": "insufficient timestamps"}
    total_s = timestamps[-1] - timestamps[0]
    expected_s = sum(iteration_counts) * result_dict.get("expected_per_iter_s", 0)
    info = {
        "phase": phase_name,
        "total_duration_s": round(total_s, 1),
        "num_iterations": len(timestamps) - 1,
        "per_iter_seconds": round(total_s / max(1, len(timestamps) - 1), 2),
    }
    if expected_s > 0:
        ratio = total_s / expected_s
        if ratio < 0.5 or ratio > 2.0:
            info["warning"] = f"total_duration ({total_s:.0f}s) vs expected ({expected_s:.0f}s): ratio={ratio:.2f}"
    return info

def integrate_power_trace(samples, window_start=None, window_end=None):
    if not samples:
        return 0.0, 0.0
    ordered = sorted(samples, key=lambda item: item[0])
    if window_start is not None:
        ordered = [item for item in ordered if item[0] >= window_start]
    if window_end is not None:
        ordered = [item for item in ordered if item[0] <= window_end]
    if not ordered:
        return 0.0, 0.0
    if len(ordered) == 1:
        if window_start is not None and window_end is not None:
            duration = window_end - window_start
            return duration * ordered[0][1], ordered[0][1]
        return 0.0, ordered[0][1]
    valid_start = window_start if window_start is not None else ordered[0][0]
    valid_end = window_end if window_end is not None else ordered[-1][0]
    energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(ordered[:-1], ordered[1:]):
        dt = t1 - t0
        if dt > 0:
            energy_j += 0.5 * (p0 + p1) * dt
    total_duration = valid_end - valid_start
    if total_duration <= 0:
        return 0.0, ordered[-1][1]
    avg_power_w = energy_j / total_duration
    return energy_j, avg_power_w


# --- Power Backends ---
class PowerBackend:
    def start(self): pass
    def stop(self, global_start_time=None, global_end_time=None): return []

class AMDGpuPowerBackend(PowerBackend):
    def __init__(self, log_file, enable_gpu_monitoring=True):
        self.log_file = log_file
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.stop_event = threading.Event()
        self.stats = []
        self.thread = None
        if not enable_gpu_monitoring:
            self.power_file = None
            self.use_rocm_smi = False
            return
        pwr_files = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power*_average") + \
                    glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power*_input")
        self.power_file = pwr_files[0] if pwr_files else None
        self.use_rocm_smi = False
        if not self.power_file:
            try:
                subprocess.run(["rocm-smi", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.use_rocm_smi = True
                print("[Info] AMD sysfs not found, fallback to 'rocm-smi' for GPU power.")
            except FileNotFoundError:
                print("[Warning] Neither sysfs nor rocm-smi found. GPU power will report as 0.")

    def _monitor(self):
        if not self.enable_gpu_monitoring or (not self.power_file and not self.use_rocm_smi):
            return
        with open(self.log_file, "w") as f_log:
            f_log.write("timestamp,gpu_power_w\n")
            sample_interval = 0.005 if self.power_file else 0.010
            while not self.stop_event.is_set():
                now = time.time()
                pwr = 0.0
                try:
                    if self.power_file:
                        with open(self.power_file, 'r') as f:
                            pwr = float(f.read().strip()) / 1e6
                    elif self.use_rocm_smi:
                        res = subprocess.check_output(["rocm-smi", "--showpower", "--json"], stderr=subprocess.DEVNULL)
                        data = json.loads(res.decode('utf-8'))
                        first_card = list(data.keys())[0]
                        pwr_str = data[first_card].get("Average Graphics Package Power (W)") or \
                                  data[first_card].get("Power (W)", "0")
                        pwr = float(pwr_str)
                except Exception:
                    pass
                if pwr > 0:
                    self.stats.append((now, pwr))
                    f_log.write(f"{now},{pwr}\n")
                time.sleep(sample_interval)

    def start(self):
        self.stats = []
        if self.enable_gpu_monitoring and (self.power_file or self.use_rocm_smi):
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def stop(self, global_start_time=None, global_end_time=None):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.stats


class RAPLBackend(PowerBackend):
    def __init__(self):
        self.supported = False
        self.path = None
        self.max_energy = None
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        energy_files = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj") +
                              glob.glob("/sys/class/powercap/amd-rapl:*/energy_uj"))
        if not energy_files:
            print("[Warning] No RAPL energy_uj found. CPU power will report as 0.")
            return
        package_path = None
        first_path = None
        for f in energy_files:
            dir_path = os.path.dirname(f)
            name_path = os.path.join(dir_path, "name")
            try:
                with open(name_path) as nf:
                    domain = nf.read().strip()
                if first_path is None:
                    first_path = f
                if "package" in domain:
                    package_path = f
                    break
            except Exception:
                continue
        self.path = package_path if package_path else (first_path if first_path else energy_files[0])
        self.supported = True
        try:
            max_path = self.path.replace("energy_uj", "max_energy_range_uj")
            with open(max_path) as fm:
                self.max_energy = int(fm.read())
        except Exception:
            self.max_energy = 2**32 - 1

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                with open(self.path) as f:
                    self.samples.append(("mid", time.time(), int(f.read())))
            except Exception:
                pass
            self.stop_event.wait(2.0)

    def start(self):
        self.samples = []
        if not self.supported:
            return
        try:
            with open(self.path) as f:
                self.samples.append(("start", time.time(), int(f.read())))
        except Exception as e:
            print(f"[Error] Failed to start RAPL monitoring: {e}")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self, global_start_time=None, global_end_time=None):
        if not self.supported:
            return 0.0, 0.0
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        try:
            with open(self.path) as f:
                self.samples.append(("stop", time.time(), int(f.read())))
        except Exception as e:
            print(f"[Error] Failed to read RAPL end energy: {e}")
        if len(self.samples) < 2:
            return 0.0, 0.0
        ordered = sorted(self.samples, key=lambda x: x[1])
        total_energy_uj = 0
        for (_, _, prev), (_, _, curr) in zip(ordered[:-1], ordered[1:]):
            delta = curr - prev
            if delta < 0:
                delta = self.max_energy - prev + curr
            total_energy_uj += delta
        if global_start_time is not None and global_end_time is not None:
            duration = global_end_time - global_start_time
        else:
            duration = ordered[-1][1] - ordered[0][1]
        if duration <= 0:
            return 0.0, 0.0
        energy_j = total_energy_uj / 1_000_000.0
        return energy_j / duration, energy_j


class AMDUProfWindowsBackend(PowerBackend):
    def __init__(self, output_dir, run_name):
        self.output_dir = output_dir
        self.run_name = run_name
        self.process = None
        self.start_epoch = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_prefix = os.path.join(self.output_dir, f"uprof_power")
        self.stats = []

    def start(self):
        self.start_epoch = time.time()
        cmd = [
            "AMDuProfCLI", "timechart",
            "--event", "power",
            "--interval", "50", 
            "-d", "999999",
            "-o", self.csv_prefix
        ]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            print(f"[Info] AMDuProfCLI started. Out to: {self.csv_prefix}*.csv")
        except Exception as e:
            print(f"[Error] Failed to start AMDuProfCLI: {e}. Is it added to PATH?")

    def stop(self, global_start_time=None, global_end_time=None):
        if self.process:
            try:
                 self.process.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                 self.process.wait(timeout=10.0)
            except Exception:
                 self.process.terminate()
            print("[Info] AMDuProfCLI stopped.")

        time.sleep(1) 
        
        search_path = os.path.join(self.csv_prefix, "*.csv")
        csv_files = glob.glob(search_path)
        if not csv_files:
            csv_files = glob.glob(self.csv_prefix + "*.csv")
            
        if not csv_files:
            print("[Warning] AMDuProfCLI did not generate any valid CSV.")
            return []

        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"[Info] Parsing uProf power log: {latest_csv}")

        self.stats = []
        try:
            with open(latest_csv, 'r') as f:
                reader = csv.reader(f)
                headers = None
                power_idx = -1
                time_idx = -1
                first_t_sec = None
                for row in reader:
                    if not row:
                        continue
                    if "RecordId" in row or "Timestamp" in row or "Time" in row:
                        headers = row
                        for i, h in enumerate(headers):
                            lower_h = h.strip().lower()
                            if "socket0-package-power" in lower_h or "package-power" in lower_h or "package_power" in lower_h or "sys_power" in lower_h:
                                power_idx = i
                                break
                        if power_idx == -1:
                            for i, h in enumerate(headers):
                                if "power" in h.lower() and "core" not in h.lower():
                                    power_idx = i
                                    break
                        if "Timestamp" in headers:
                            time_idx = headers.index("Timestamp")
                        elif "Time" in headers:
                            time_idx = headers.index("Time")
                        else:
                            time_idx = 1
                    elif headers and power_idx != -1 and time_idx != -1:
                        try:
                            t_str = row[time_idx].strip()
                            parts = t_str.split(':')
                            
                            if len(parts) == 4:
                                h, m, s, ms = map(float, parts)
                                t_sec = h * 3600 + m * 60 + s + ms / 1000.0
                            else:
                                t_sec = float(t_str) / 1000.0 

                            if first_t_sec is None:
                                first_t_sec = t_sec
                            
                            pwr = float(row[power_idx].strip())
                            abs_time = self.start_epoch + (t_sec - first_t_sec)
                            self.stats.append((abs_time, pwr))
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[Error] Failed to parse AMDuProf output: {e}")

        return self.stats


class RAPLWindowsBackend(PowerBackend):
    def __init__(self):
        self.stats = []
        self.stop_event = threading.Event()
        self.thread = None
        self._prewarm()

    def _prewarm(self):
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Power' -SampleInterval 1 -MaxSamples 1"],
                capture_output=True, text=True, timeout=10
            )
        except Exception:
            pass

    def _sample_power_w(self):
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Power').CounterSamples.CookedValue"],
                capture_output=True, text=True, timeout=5
            )
            val = float(result.stdout.strip())
            return val / 1000.0
        except subprocess.TimeoutExpired:
            print("[Warning] Get-Counter timed out after 5s")
            return 0.0
        except ValueError:
            print(f"[Warning] Failed to parse Get-Counter output: {result.stdout.strip()!r}")
            return 0.0
        except FileNotFoundError:
            print("[Warning] PowerShell not found. Cannot read Energy Meter.")
            return 0.0
        except Exception as e:
            print(f"[Warning] Get-Counter failed: {type(e).__name__}: {e}")
            return 0.0

    def _monitor(self):
        while not self.stop_event.is_set():
            pwr = self._sample_power_w()
            now = time.time()
            self.stats.append((now, pwr))
            if pwr == 0:
                print(f"[Debug] Get-Counter returned 0W at t={now:.3f}")
            time.sleep(0.1)

    def start(self):
        self.stats = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self, global_start_time=None, global_end_time=None):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        return self.stats


class UnifiedPowerMonitor:
    def __init__(self, output_dir, run_name, os_type, is_integrated=False):
        self.os_type = os_type
        self.is_integrated = is_integrated
        
        if self.os_type == 'linux':
            hw_pwr_log = get_timestamped_filename(output_dir, f"{run_name}_power_hw", "csv")
            self.gpu_backend = AMDGpuPowerBackend(hw_pwr_log, enable_gpu_monitoring=not is_integrated)
            self.cpu_backend = RAPLBackend()
            self.hw_log_file = hw_pwr_log
        else:
            self.backend = RAPLWindowsBackend()

    def start(self):
        if self.os_type == 'linux':
            self.gpu_backend.start()
            self.cpu_backend.start()
        else:
            print("[Info] Using Windows RAPL Energy Meter for power monitoring.")
            self.backend.start()

    def stop(self, global_start_time=None, global_end_time=None):
        if self.os_type == 'linux':
            gpu_stats = self.gpu_backend.stop()
            cpu_avg_w, cpu_energy_j = self.cpu_backend.stop(global_start_time, global_end_time)
            gpu_energy_j = 0.0
            avg_gpu_w = 0.0
            if not self.is_integrated and gpu_stats:
                gpu_energy_j, avg_gpu_w = integrate_power_trace(gpu_stats, global_start_time, global_end_time)
            if self.is_integrated:
                total_energy_j = cpu_energy_j
                measurement_mode = "RAPL_only_integrated_gpu"
            else:
                total_energy_j = gpu_energy_j + cpu_energy_j
                measurement_mode = "gpu_plus_cpu_separate"
            return {
                "avg_gpu_w": avg_gpu_w, 
                "avg_cpu_w": cpu_avg_w,
                "avg_total_w": (total_energy_j / (global_end_time - global_start_time)) if global_end_time > global_start_time else 0.0,
                "gpu_energy_j": gpu_energy_j, 
                "cpu_energy_j": cpu_energy_j,
                "total_energy_j": total_energy_j,
                "gpu_stats": gpu_stats if not self.is_integrated else [],
                "measurement_mode": measurement_mode,
                "is_integrated": self.is_integrated,
                "hw_log_file": self.hw_log_file
            }
        else:
            stats = self.backend.stop(global_start_time, global_end_time)
            energy_j, avg_w = integrate_power_trace(stats, global_start_time, global_end_time)
            
            return {
                "avg_total_w": avg_w,
                "total_energy_j": energy_j,
                "stats": stats,
                "measurement_mode": "RAPL_Windows",
                "is_integrated": self.is_integrated
            }

    def measure_baseline(self, duration_s=10.0):
        if self.os_type == 'linux':
            rapl = RAPLBackend()
            rapl.start()
            time.sleep(duration_s)
            avg_w, _ = rapl.stop()
            return avg_w
        else:
            backend = RAPLWindowsBackend()
            backend.start()
            time.sleep(duration_s)
            stats = backend.stop()
            _, avg_w = integrate_power_trace(stats)
            return avg_w


def wslpath_a(win_path):
    win_path = win_path.replace("\\", "/")
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        return "/mnt/" + drive + win_path[2:]
    return win_path

def run_wsl_host(config, args):
    print("--- Starting WSL Orchestration (Host side on Windows) ---")
    base_output_dir = config.get("output_dir", "./profiling_logs/benchmarker")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.abspath(os.path.join(base_output_dir, f"run_{ts}_wsl"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    power_monitor = RAPLWindowsBackend()
    power_monitor.start()

    wsl_config = config.copy()
    wsl_config["os_type"] = "wsl"

    wsl_config_path = os.path.join(OUTPUT_DIR, "wsl_config.json")
    with open(wsl_config_path, "w") as f:
        json.dump(wsl_config, f, indent=4)

    script_path = os.path.abspath(__file__)

    wsl_script_path = wslpath_a(script_path)
    wsl_config_wsl_path = wslpath_a(wsl_config_path)
    wsl_output_dir_wsl_path = wslpath_a(OUTPUT_DIR)

    distro = config.get("wsl", {}).get("distro", "")
    python_path = config.get("wsl", {}).get("python_path", "python3")

    cmd = ["wsl"]
    if distro:
        cmd.extend(["-d", distro])
    cmd.extend(["-u", config.get("wsl", {}).get("user", "user")])

    bash_cmd = f"{python_path} {wsl_script_path} --config {wsl_config_wsl_path} --wsl-worker --wsl-output-dir {wsl_output_dir_wsl_path}"
    cmd.extend(["--", "bash", "-ic", bash_cmd])

    print(f"[Info] Dispatching task to WSL...")
    print(f"[Info] Command: {' '.join(cmd)}")
    subprocess.run(cmd)

    stats = power_monitor.stop()

    idle_monitor = RAPLWindowsBackend()
    idle_monitor.start()
    time.sleep(10.0)
    idle_stats = idle_monitor.stop()
    _, idle_power = integrate_power_trace(idle_stats)

    wsl_out_json_path = os.path.join(OUTPUT_DIR, "wsl_worker_metrics.json")
    if os.path.exists(wsl_out_json_path):
        with open(wsl_out_json_path, "r") as f:
            wsl_results = json.load(f)

        for res in wsl_results:
            if "wsl_pwr_start_epoch" in res and "wsl_pwr_end_epoch" in res:
                start_p = res["wsl_pwr_start_epoch"]
                end_p = res["wsl_pwr_end_epoch"]
                energy_j, avg_w = integrate_power_trace(stats, start_p, end_p)
                pwr_duration_s = end_p - start_p
                energy_per_step_j = energy_j / max(1, config["execution"]["passes"]["power"]["num_test"])
                avg_w_adjusted = max(0, avg_w - idle_power)
                tokens_per_j = config["output"]["output_tokens"] / energy_per_step_j if energy_per_step_j > 0 else 0

                res["power_energy"]["avg_total_adjusted_w"] = avg_w_adjusted
                res["power_energy"]["avg_idle_w"] = round(idle_power, 2)
                res["power_energy"]["energy_per_inference_j"] = energy_per_step_j
                res["power_energy"]["tokens_per_joule_p50"] = tokens_per_j
                res["power_energy"]["raw_energy_j"] = energy_j
                res["power_energy"]["measurement_mode"] = "RAPL_Windows_WSL"
                res["power_energy"]["note"] = f"Integrated from Host (dur: {pwr_duration_s:.2f}s, idle: {idle_power:.2f}W)"

        metrics_json = get_timestamped_filename(OUTPUT_DIR, "benchmarker_metrics_wsl_final", "json")
        with open(metrics_json, "w") as f:
            json.dump(wsl_results, f, indent=4)
        print(f"\nFull integrated WSL metrics saved to: {metrics_json}")
    else:
        print("[Error] WSL worker JSON not found. Worker may have crashed.")


def get_cpu_mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def analyze_memory_details(device_name, model):
    mem_details = {}
    if device_name == "cuda":
        try:
            mem_details["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            mem_details["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            mem_details["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            mem_details["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            mem_details["model_params_mb"] = param_size
        except:
            pass
    else:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        mem_details["cpu_rss_mb"] = rss_mb
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            mem_details["model_params_mb"] = param_size
        except:
            pass
    return mem_details

def get_valid_configs(config):
    configs = []
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    device_cfg = config["execution"].get("device", "cuda")
    devices = device_cfg if isinstance(device_cfg, list) else [device_cfg]
    selected_dtypes = [dtype_map.get(d, torch.bfloat16) for d in config["execution"]["dtype"]]
    gpu_options = config["execution"].get("gpu_options", {})
    attn_impls = gpu_options.get("attn_implementation", ["sdpa"])
    cpu_options = config["execution"].get("cpu_options", {})
    cpu_accs = cpu_options.get("cpu_acc", ["none"])
    for dev in devices:
        for dt in selected_dtypes:
            if dev == "cpu":
                for acc in cpu_accs:
                    configs.append({"device": dev, "dtype": dt, "attn": "sdpa", "cpu_acc": acc})
            else:
                for attn in attn_impls:
                    if dt == torch.float32 and attn == "flash_attention_2": continue
                    configs.append({"device": dev, "dtype": dt, "attn": attn, "cpu_acc": "none"})
    return configs

def profile_smolvlm(config, args):
    check_environment(config, args)
    os_type = config.get("os_type", "linux")
    is_wsl_worker = getattr(args, 'wsl_worker', False)

    if is_wsl_worker and getattr(args, 'wsl_output_dir', None):
        OUTPUT_DIR = getattr(args, 'wsl_output_dir')
    else:
        base_output_dir = config.get("output_dir", "./profiling_logs/benchmarker")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(base_output_dir, f"run_{ts}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_ID = config["model"]
    MAX_NEW_TOKENS = config["output"]["output_tokens"]
    NUM_WARMUP = config["execution"]["passes"]["warmup"]["num_warmup"]
    NUM_TEST_LATENCY = config["execution"]["passes"]["end_to_end"]["num_test"]
    NUM_TEST_POWER = config["execution"]["passes"]["power"]["num_test"]
    TEMP = config["execution"]["temperature"]
    DO_SAMPLE = config["execution"]["sampling_randomness"]
    EFFECTIVE_TEMPERATURE = TEMP if TEMP is not None else (1.0 if DO_SAMPLE else None)
    DO_BW_ANALYSIS = config.get("target_metrics", {}).get("bandwidth_analysis", False)

    configs = get_valid_configs(config)
    if not configs:
        print("No valid configurations to run. Check your arguments.")
        return

    mode_str = "WSL Worker" if is_wsl_worker else os_type
    print(f"--- Starting Profiling for {MODEL_ID} on {mode_str} ---")
    print(f"Testing {len(configs)} configuration(s)...")

    inputs_cfg = config.get("inputs", {})
    image_paths = inputs_cfg.get("images", [])
    if image_paths:
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]
        num_images = len(images)
    else:
        image_size = inputs_cfg.get("image_size", 384)
        num_images = inputs_cfg.get("num_images", 2)
        images = [ToPILImage()(torch.ones(3, image_size, image_size)) for _ in range(num_images)]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt_text = inputs_cfg.get("prompt", "")
    prompt_size = inputs_cfg.get("prompt_size", 0)
    if not prompt_text and prompt_size > 0:
        prompt_text = "A"
    prompt_content = [{"type": "image"} for _ in range(num_images)]
    if prompt_text:
        prompt_content.append({"type": "text", "text": prompt_text})
    prompt = processor.apply_chat_template([{"role": "user", "content": prompt_content}], add_generation_prompt=True)

    all_results = []

    for cfg in configs:
        dev_name = cfg["device"]
        dtype = cfg["dtype"]
        attn = cfg["attn"]
        cpu_acc = cfg.get("cpu_acc", "none")
        if dev_name == "cpu":
            run_name = f"{dev_name}_{str(dtype).split('.')[-1]}_{cpu_acc}"
        else:
            run_name = f"{dev_name}_{str(dtype).split('.')[-1]}_{attn}"

        print(f"\n{'=' * 45}")
        print(f"  Profiling Config: {run_name}")
        print(f"{'=' * 45}")

        try:
            device = torch.device(dev_name)
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, torch_dtype=dtype, attn_implementation=attn,
            ).to(device).eval()

            inputs = processor(text=prompt, images=images, return_tensors="pt")

            if prompt_size > 0:
                seq_len = inputs["input_ids"].shape[1]
                if seq_len < prompt_size:
                    pad_len = prompt_size - seq_len
                    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
                    pads = torch.full((1, pad_len), pad_id, dtype=inputs["input_ids"].dtype)
                    inputs["input_ids"] = torch.cat([pads, inputs["input_ids"]], dim=1)
                    if "attention_mask" in inputs:
                        mask_pads = torch.zeros((1, pad_len), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device)
                        inputs["attention_mask"] = torch.cat([mask_pads, inputs["attention_mask"]], dim=1)

            inputs = {
                k: v.to(device=device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in inputs.items()
            }

            sanity = {}

            # === Warmup ===
            t_warmup_0 = time.perf_counter()
            print(f"Warmup: {NUM_WARMUP} iterations...")
            with torch.no_grad():
                for i in range(NUM_WARMUP):
                    print(f"  Warmup {i+1}/{NUM_WARMUP}...", end="\r", flush=True)
                    wk = dict(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=DO_SAMPLE)
                    if EFFECTIVE_TEMPERATURE is not None:
                        wk["temperature"] = EFFECTIVE_TEMPERATURE
                    _ = model.generate(**inputs, **wk)
            print()
            t_warmup_1 = time.perf_counter()
            sanity["warmup_duration_s"] = round(t_warmup_1 - t_warmup_0, 1)

            # === Memory baseline ===
            cpu_mem_start = get_cpu_mem_gb()
            if dev_name == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # === Latency with Streamer ===
            ttft_list, tpot_list = [], []
            iter_timestamps = [time.perf_counter()]
            print(f"Latency: {NUM_TEST_LATENCY} iterations...")
            with torch.no_grad():
                for i in range(NUM_TEST_LATENCY):
                    print(f"  Latency run {i+1}/{NUM_TEST_LATENCY}...", end="\r", flush=True)
                    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True)
                    gen_kwargs = dict(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=MAX_NEW_TOKENS,
                        use_cache=True,
                        do_sample=DO_SAMPLE,
                        streamer=streamer,
                    )
                    if EFFECTIVE_TEMPERATURE is not None:
                        gen_kwargs["temperature"] = EFFECTIVE_TEMPERATURE

                    if dev_name == "cuda":
                        torch.cuda.synchronize()
                    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)

                    ttft = None
                    token_times = []
                    t_start = time.perf_counter()
                    thread.start()
                    for _ in streamer:
                        t_now = time.perf_counter()
                        if ttft is None:
                            ttft = (t_now - t_start) * 1000
                        else:
                            token_times.append((t_now - t_prev) * 1000)
                        t_prev = t_now
                    thread.join()

                    ttft_list.append(ttft)
                    tpot_values = token_times if len(token_times) > 1 else [0]
                    tpot_list.append(np.mean(tpot_values))
                    iter_timestamps.append(time.perf_counter())

            per_iter_s = [(iter_timestamps[i+1] - iter_timestamps[i]) for i in range(len(iter_timestamps)-1)]
            median_iter_s = float(np.median(per_iter_s))
            p95_iter_s = float(np.percentile(per_iter_s, 95))
            sanity["latency"] = {
                "total_s": round(iter_timestamps[-1] - iter_timestamps[0], 1),
                "per_iter_median_s": round(median_iter_s, 2),
                "per_iter_p95_s": round(p95_iter_s, 2),
                "iter_outliers": [i for i, t in enumerate(per_iter_s) if t > 2 * median_iter_s]
            }

            # === Memory ===
            if dev_name == "cuda":
                peak_mem_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
                peak_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            else:
                cpu_mem_end = get_cpu_mem_gb()
                peak_mem_allocated_gb = max(cpu_mem_end - cpu_mem_start, 0.1)
                peak_mem_reserved_gb = peak_mem_allocated_gb
            mem_details = analyze_memory_details(dev_name, model)

            # === Power ===
            is_integrated = config["execution"].get("is_integrated", False)
            
            if is_wsl_worker:
                print(f"Power: {NUM_TEST_POWER} iterations (WSL Worker Mode)...")
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t_pwr_start = time.time()
                with torch.no_grad():
                    for i in range(NUM_TEST_POWER):
                        print(f"  Power run {i+1}/{NUM_TEST_POWER}...", end="\r", flush=True)
                        pk = dict(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=DO_SAMPLE)
                        if EFFECTIVE_TEMPERATURE is not None:
                            pk["temperature"] = EFFECTIVE_TEMPERATURE
                        _ = model.generate(**inputs, **pk)
                print()
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t_pwr_end = time.time()
                
                metrics_data = {
                    "total_energy_j": 0, "avg_gpu_w": 0, "avg_cpu_w": 0,
                    "measurement_mode": "Waiting_for_Host_WSL",
                    "gpu_energy_j": 0, "cpu_energy_j": 0
                }
                avg_power = 0
                total_energy_j = 0
                
            else:
                monitor = UnifiedPowerMonitor(OUTPUT_DIR, run_name, os_type, is_integrated=is_integrated)
                pwr_mode_str = "RAPL" if os_type == "linux" else "RAPL_Windows"
                print(f"Power: {NUM_TEST_POWER} iterations (Using {pwr_mode_str})...")

                if dev_name == "cuda":
                    torch.cuda.synchronize()
                monitor.start()
                t_pwr_start = time.time()
                
                with torch.no_grad():
                    for i in range(NUM_TEST_POWER):
                        print(f"  Power run {i+1}/{NUM_TEST_POWER}...", end="\r", flush=True)
                        pk = dict(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=DO_SAMPLE)
                        if EFFECTIVE_TEMPERATURE is not None:
                            pk["temperature"] = EFFECTIVE_TEMPERATURE
                        _ = model.generate(**inputs, **pk)
                        if dev_name == "cuda":
                            torch.cuda.synchronize()
                print()
                        
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t_pwr_end = time.time()
                
                metrics_data = monitor.stop(global_start_time=t_pwr_start, global_end_time=t_pwr_end)
                total_energy_j = metrics_data["total_energy_j"]
                avg_power = metrics_data.get("avg_total_w", 0)
                idle_power = monitor.measure_baseline(duration_s=10.0)

            avg_power_adjusted = max(0, avg_power - idle_power) if not is_wsl_worker else avg_power
            pwr_duration_s = t_pwr_end - t_pwr_start
            
            # === Bandwidth Verification ===
            bw_result = {}

            ttft_stats = calculate_metrics(ttft_list)
            tpot_stats = calculate_metrics(tpot_list)
            energy_per_step_j = total_energy_j / max(1, NUM_TEST_POWER) if not is_wsl_worker else 0
            tokens_per_j = MAX_NEW_TOKENS / energy_per_step_j if energy_per_step_j > 0 else 0

            power_struct = {
                "avg_total_adjusted_w": avg_power_adjusted,
                "avg_idle_w": round(idle_power, 2) if not is_wsl_worker else 0,
                "energy_per_inference_j": energy_per_step_j,
                "tokens_per_joule_p50": tokens_per_j,
                "raw_energy_j": total_energy_j,
                "measurement_mode": metrics_data.get("measurement_mode", "unknown")
            }
            if os_type == "linux" and not is_wsl_worker:
                power_struct["avg_gpu_w"] = metrics_data.get("avg_gpu_w", 0.0)
                power_struct["avg_cpu_w"] = metrics_data.get("avg_cpu_w", 0.0)
                power_struct["raw_gpu_energy_j"] = metrics_data.get("gpu_energy_j", 0.0)
                power_struct["raw_cpu_energy_j"] = metrics_data.get("cpu_energy_j", 0.0)

            res = {
                "config": run_name,
                "status": "ok",
                "is_integrated": is_integrated,
                "measurement_mode": metrics_data.get("measurement_mode", "unknown"),
                "latency": {
                    "ttft_ms_mean": ttft_stats["mean"], "ttft_ms_p50": ttft_stats["median"],
                    "ttft_ms_p95": ttft_stats["p95"], "tpot_ms_mean": tpot_stats["mean"],
                    "tpot_ms_p50": tpot_stats["median"], "tpot_ms_p95": tpot_stats["p95"],
                },
                "power_energy": power_struct,
                "memory": {
                    "peak_mem_allocated_gb": peak_mem_allocated_gb,
                    "peak_mem_reserved_gb": peak_mem_reserved_gb,
                    **mem_details,
                },
                "bandwidth_analysis": bw_result,
                "sanity": sanity,
                "timestamp": datetime.datetime.now().isoformat()
            }
            if is_wsl_worker:
                res["wsl_pwr_start_epoch"] = t_pwr_start
                res["wsl_pwr_end_epoch"] = t_pwr_end
                
            if "hw_log_file" in metrics_data:
                res["hw_log_file"] = metrics_data["hw_log_file"]

            all_results.append(res)
            print(f"Results: TTFT {ttft_stats['mean']:.0f}ms | TPOT {tpot_stats['mean']:.1f}ms/tok")

            del model
            torch.cuda.empty_cache() if dev_name == "cuda" else None
            gc.collect()

        except Exception as e:
            all_results.append({
                "config": run_name,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            })
            print(f"Failed config {run_name}: {type(e).__name__}: {e}")
            if 'model' in locals():
                del model
            if dev_name == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    if is_wsl_worker:
        metrics_json = os.path.join(OUTPUT_DIR, "wsl_worker_metrics.json")
    else:
        metrics_json = get_timestamped_filename(OUTPUT_DIR, "benchmarker_metrics", "json")
        
    with open(metrics_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nFull metrics saved to: {metrics_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Vision Autoregressive Models with dynamic metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file (required).")
    parser.add_argument("--wsl-worker", action="store_true", help="Internal flag. Do not use directly.")
    parser.add_argument("--wsl-output-dir", type=str, help="Output directory when in wsl-worker mode.")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        sys.exit(f"[Error] Config file not found: {args.config}")
    
    with open(args.config, "r") as f:
        user_config = json.load(f)
    
    config = dict_deep_update(DEFAULT_CONFIG.copy(), user_config)
    profile_smolvlm(config, args)
