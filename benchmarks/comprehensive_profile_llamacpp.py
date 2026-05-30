import time
import json
import os
import sys
import threading
import subprocess
import numpy as np
import psutil
import argparse
import glob
import datetime
import gc
import csv
import base64
import socket
import urllib.request
import urllib.error


DEFAULT_CONFIG = {
    "os_type": "linux",
    "task_type": "vision_autoregressive",
    "backend": "llamacpp",
    "model": "SmolVLM-Instruct",
    "output_dir": "./profiling_logs/llamacpp",
    "models_root": "../models",
    "llamacpp": {
        "dir": "",
        "ngl": None,
        "threads": None,
        "flash_attn": "auto",
    },
    "execution": {
        "device": "cpu",
        "quantizations": ["f16", "Q8_0", "Q4_K_M"],
        "is_integrated": True,
        "sampling_randomness": False,
        "temperature": None,
        "passes": {
            "warmup": {"num_warmup": 10},
            "end_to_end": {"num_test": 20},
            "power": {"num_test": 10}
        }
    },
    "inputs": {
        "prompt": "Describe the images briefly.",
        "prompt_size": 64,
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
        "bandwidth_analysis": False
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
    if not data:
        return {}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "std": float(np.std(data))
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
        cmd = ["AMDuProfCLI", "timechart", "--event", "power", "--interval", "50", "-d", "999999", "-o", self.csv_prefix]
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
                capture_output=True, text=True, timeout=10)
        except Exception:
            pass

    def _sample_power_w(self):
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Power').CounterSamples.CookedValue"],
                capture_output=True, text=True, timeout=5)
            val = float(result.stdout.strip())
            return val / 1000.0
        except Exception:
            return 0.0

    def _monitor(self):
        while not self.stop_event.is_set():
            pwr = self._sample_power_w()
            self.stats.append((time.time(), pwr))
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


# --- llama-server helpers ---
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_llama_server(llamacpp_dir, model_path, mmproj_path, port, ngl=None, threads=None, flash_attn="auto"):
    cmd = [os.path.join(llamacpp_dir, "llama-server")]
    cmd += ["-m", model_path, "--mmproj", mmproj_path]
    cmd += ["--host", "127.0.0.1", "--port", str(port), "--no-mmap"]
    cmd += ["--no-warmup"]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if flash_attn:
        cmd += ["--flash-attn", flash_attn]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = llamacpp_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    return proc


def wait_for_server_ready(port, timeout=120):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except urllib.error.URLError as e:
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    print(f"[Error] Server not ready after {timeout}s on port {port}: {last_err}")
    return False


def stop_llama_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def encode_images_to_base64(image_paths):
    """Encode images to base64 strings for llama.cpp multimodal API.
    
    llama-server expects a flat array of base64 strings for multimodal_data,
    and the prompt must contain media_marker placeholders (e.g., "<__media__>").
    """
    encoded = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            encoded.append(b64)
    return encoded


def get_media_marker(port):
    """Get the dynamic media_marker from llama-server /props endpoint.
    
    llama-server generates a unique media_marker per session that must be
    used in the prompt to indicate where image embeddings should be inserted.
    """
    url = f"http://127.0.0.1:{port}/props"
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        props = json.loads(resp.read().decode("utf-8"))
        return props.get("media_marker", "")
    except Exception as e:
        print(f"[Warning] Failed to get media_marker: {e}")
        return ""


def call_completion(port, prompt, multimodal_data, n_predict, temperature=None):
    """Call llama-server /completion endpoint with multimodal support.
    
    Args:
        port: Server port
        prompt: Text prompt containing media_marker placeholders (e.g., "<__media__> Describe...")
        multimodal_data: List of base64-encoded image strings
        n_predict: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        (result_dict, success_bool)
    """
    url = f"http://127.0.0.1:{port}/completion"
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "min_tokens": n_predict,  # Force minimum tokens to ensure full generation
        "multimodal_data": multimodal_data,
        "temperature": temperature if temperature is not None else 0.0,
        "cache_prompt": False,
        "ignore_eos": True,  # Prevent early stopping on EOS
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        result = json.loads(resp.read().decode("utf-8"))
        return result, True
    except Exception as e:
        return {"error": str(e)}, False


def build_multimodal_prompt(prompt_text, media_marker, num_images):
    """Build a prompt with media_marker placeholders for llama.cpp multimodal API.
    
    llama-server requires the media_marker (obtained from /props) to be placed
    in the prompt where image embeddings should be inserted.
    """
    placeholders = media_marker * num_images
    return f"{placeholders} {prompt_text}"


def extract_timings(completion_response):
    timings = completion_response.get("timings", {})
    ttft_ms = timings.get("prompt_ms", 0)
    tpot_ms = timings.get("predicted_per_token_ms", 0)
    predicted_n = timings.get("predicted_n", 0)
    total_ms = timings.get("predicted_ms", 0) + timings.get("prompt_ms", 0)
    return ttft_ms, tpot_ms, predicted_n, total_ms


# --- Model discovery ---
def find_gguf_variants(models_root, model_name):
    model_dir = os.path.join(models_root, f"{model_name}-GGUF")
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(models_root, model_name)
    if not os.path.isdir(model_dir):
        return []
    mmprojs = sorted(glob.glob(os.path.join(model_dir, "mmproj-*.gguf")))
    ggufs = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    text_models = [f for f in ggufs if not os.path.basename(f).startswith("mmproj-")]
    if not text_models or not mmprojs:
        return []
    results = []
    for model_path in text_models:
        base = os.path.basename(model_path).replace(".gguf", "")
        quant = base.replace(f"{model_name}-", "") if model_name in base else base
        results.append({
            "model_name": model_name,
            "model_path": model_path,
            "mmproj_path": mmprojs[0],
            "quant": quant if quant and quant != base else "unknown",
        })
    return results


def get_cpu_mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def analyze_memory_details():
    return {"cpu_rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024**2)}


def pad_prompt(prompt, prompt_size):
    if prompt_size <= 0:
        return prompt
    rough_tokens = len(prompt.split())
    if rough_tokens >= prompt_size:
        return prompt
    pad_needed = prompt_size - rough_tokens
    filler = "A " * (pad_needed // 2 + 1)
    return filler[:pad_needed * 2] + prompt


def generate_synthetic_image(output_path, image_size=384, num_images=2):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing = []
    for i in range(num_images):
        path = output_path.replace(".png", f"_{i}.png")
        if os.path.exists(path):
            existing.append(path)
    if len(existing) == num_images:
        return existing
    from PIL import Image
    import numpy as np
    paths = []
    for i in range(num_images):
        path = output_path.replace(".png", f"_{i}.png")
        arr = (np.ones((image_size, image_size, 3), dtype=np.uint8) * 128)
        Image.fromarray(arr, mode="RGB").save(path)
        print(f"[Info] Generated synthetic image: {path}")
        paths.append(path)
    return paths


def get_valid_configs(config):
    configs = []
    device_cfg = config["execution"].get("device", "cpu")
    devices = device_cfg if isinstance(device_cfg, list) else [device_cfg]
    quants_cfg = config["execution"].get("quantizations", ["f16"])
    model_name = config["model"]
    MODELS_ROOT = config.get("models_root")
    variants = find_gguf_variants(MODELS_ROOT, model_name)
    if not variants:
        print(f"[Warning] No GGUF files found for model '{model_name}' in {MODELS_ROOT}")
        return configs
    for dev in devices:
        for v in variants:
            if v["quant"] not in quants_cfg:
                continue
            configs.append({
                "device": dev,
                "model_path": v["model_path"],
                "mmproj_path": v["mmproj_path"],
                "model_name": model_name,
                "quant": v["quant"],
            })
    return configs


# --- Main profiling ---
def profile_llamacpp(config, args):
    os_type = config.get("os_type", "linux")
    is_wsl_worker = getattr(args, 'wsl_worker', False)

    if is_wsl_worker and getattr(args, 'wsl_output_dir', None):
        OUTPUT_DIR = getattr(args, 'wsl_output_dir')
    else:
        base_output_dir = config.get("output_dir", "./profiling_logs/llamacpp")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(base_output_dir, f"run_{ts}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_ID = config["model"]
    LLAMACPP_DIR = config["llamacpp"]["dir"]
    MAX_NEW_TOKENS = config["output"]["output_tokens"]
    NUM_WARMUP = config["execution"]["passes"]["warmup"]["num_warmup"]
    NUM_TEST_LATENCY = config["execution"]["passes"]["end_to_end"]["num_test"]
    NUM_TEST_POWER = config["execution"]["passes"]["power"]["num_test"]
    TEMP = config["execution"]["temperature"]
    DO_SAMPLE = config["execution"]["sampling_randomness"]
    EFFECTIVE_TEMPERATURE = TEMP if TEMP is not None else (1.0 if DO_SAMPLE else None)
    DO_BW_ANALYSIS = config.get("target_metrics", {}).get("bandwidth_analysis", False)
    NGL = config["llamacpp"].get("ngl")
    THREADS = config["llamacpp"].get("threads")
    FLASH_ATTN = config["llamacpp"].get("flash_attn", "auto")

    if not LLAMACPP_DIR:
        sys.exit("[Error] llamacpp.dir not set in config")
    if not os.path.isdir(LLAMACPP_DIR):
        sys.exit(f"[Error] llamacpp.dir not found: {LLAMACPP_DIR}")
    if not os.path.isfile(os.path.join(LLAMACPP_DIR, "llama-server")):
        sys.exit(f"[Error] llama-server not found in {LLAMACPP_DIR}")

    inputs_cfg = config.get("inputs", {})
    image_paths = inputs_cfg.get("images", [])
    image_size = inputs_cfg.get("image_size", 384)
    num_images = inputs_cfg.get("num_images", 2)
    if image_paths:
        image_paths_used = image_paths
    else:
        synthetic_dir = os.path.expanduser("~/.cache/llamacpp_profiling")
        synthetic_base = os.path.join(synthetic_dir, f"synth_img_{image_size}.png")
        image_paths_used = generate_synthetic_image(synthetic_base, image_size=image_size, num_images=num_images)

    prompt_text = inputs_cfg.get("prompt", "Describe the images briefly.")
    prompt_size = inputs_cfg.get("prompt_size", 0)
    padded_prompt = pad_prompt(prompt_text, prompt_size)
    if padded_prompt != prompt_text:
        print(f"[Info] Padded prompt from ~{len(prompt_text.split())} to ~{prompt_size} tokens")

    # Encode images once
    image_data = encode_images_to_base64(image_paths_used)

    configs = get_valid_configs(config)
    if not configs:
        print("No valid configurations to run. Check your arguments.")
        return

    mode_str = "WSL Worker" if is_wsl_worker else os_type
    print(f"--- Starting Profiling for {MODEL_ID} on {mode_str} via llama-server ---")
    print(f"Testing {len(configs)} configuration(s)...")

    all_results = []

    for cfg in configs:
        dev_name = cfg["device"]
        quant = cfg["quant"]
        run_name = f"{dev_name}_{quant}"

        print(f"\n{'=' * 45}")
        print(f"  Profiling Config: {run_name}")
        print(f"  Model: {cfg['model_path']}")
        print(f"{'=' * 45}")

        port = find_free_port()
        server_proc = None

        try:
            print(f"  Starting llama-server on port {port}...")
            server_proc = start_llama_server(
                LLAMACPP_DIR, cfg["model_path"], cfg["mmproj_path"], port,
                ngl=NGL, threads=THREADS, flash_attn=FLASH_ATTN,
            )
            if not wait_for_server_ready(port):
                raise RuntimeError("llama-server failed to start")

            # Get the dynamic media_marker from server for multimodal prompts
            media_marker = get_media_marker(port)
            if not media_marker:
                raise RuntimeError("Failed to get media_marker from server - multimodal may not work")
            print(f"  Media marker: {media_marker[:30]}...")

            # Build multimodal prompt with correct media_marker placeholders
            multimodal_prompt = build_multimodal_prompt(padded_prompt, media_marker, num_images)

            sanity = {}

            # === Warmup ===
            t_warmup_0 = time.perf_counter()
            print(f"  Warmup: {NUM_WARMUP} iterations...")
            for i in range(NUM_WARMUP):
                print(f"    Warmup {i+1}/{NUM_WARMUP}...", end="\r", flush=True)
                resp, ok = call_completion(port, multimodal_prompt, image_data, MAX_NEW_TOKENS, EFFECTIVE_TEMPERATURE)
                if not ok:
                    print(f"\n    Warmup {i+1}: FAILED - {resp.get('error', '')}")
            print()
            t_warmup_1 = time.perf_counter()
            sanity["warmup_duration_s"] = round(t_warmup_1 - t_warmup_0, 1)

            # === Memory baseline ===
            # Note: We measure the llama-server subprocess memory, not the Python script
            server_pid = server_proc.pid
            server_proc_mem = psutil.Process(server_pid) if server_pid else None

            # === Latency ===
            ttft_list, tpot_list = [], []
            iter_timestamps = [time.perf_counter()]
            print(f"  Latency: {NUM_TEST_LATENCY} iterations...")
            for i in range(NUM_TEST_LATENCY):
                resp, ok = call_completion(port, multimodal_prompt, image_data, MAX_NEW_TOKENS, EFFECTIVE_TEMPERATURE)
                if not ok:
                    print(f"    Run {i+1}/{NUM_TEST_LATENCY}: FAILED - {resp.get('error', '')}")
                    continue
                ttft_ms, tpot_ms, predicted_n, total_ms = extract_timings(resp)
                ttft_list.append(ttft_ms)
                tpot_list.append(tpot_ms)
                iter_timestamps.append(time.perf_counter())
                print(f"    Run {i+1}/{NUM_TEST_LATENCY}: TTFT={ttft_ms:.0f}ms, TPOT={tpot_ms:.1f}ms/tok, tokens={predicted_n}")

            per_iter_s = [(iter_timestamps[j+1] - iter_timestamps[j]) for j in range(len(iter_timestamps)-1)]
            median_iter_s = float(np.median(per_iter_s)) if per_iter_s else 0
            p95_iter_s = float(np.percentile(per_iter_s, 95)) if per_iter_s else 0
            sanity["latency"] = {
                "total_s": round(iter_timestamps[-1] - iter_timestamps[0], 1) if len(iter_timestamps) > 1 else 0,
                "per_iter_median_s": round(median_iter_s, 2),
                "per_iter_p95_s": round(p95_iter_s, 2),
                "iter_outliers": [i for i, t in enumerate(per_iter_s) if t > 2 * median_iter_s]
            }

            # === Memory ===
            # Measure llama-server subprocess memory (not the Python orchestrator)
            if server_proc_mem:
                try:
                    mem_info = server_proc_mem.memory_info()
                    cpu_rss_mb = mem_info.rss / (1024**2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu_rss_mb = 0.0
            else:
                cpu_rss_mb = 0.0

            peak_mem_allocated_gb = round(cpu_rss_mb / 1024, 2)
            peak_mem_reserved_gb = peak_mem_allocated_gb
            mem_details = {"cpu_rss_mb": round(cpu_rss_mb, 2)}

            # === Power ===
            is_integrated = config["execution"].get("is_integrated", True)

            if is_wsl_worker:
                print(f"  Power: {NUM_TEST_POWER} iterations (WSL Worker Mode)...")
                t_pwr_start = time.time()
                for i in range(NUM_TEST_POWER):
                    print(f"    Power run {i+1}/{NUM_TEST_POWER}...", end="\r", flush=True)
                    call_completion(port, multimodal_prompt, image_data, MAX_NEW_TOKENS, EFFECTIVE_TEMPERATURE)
                print()
                t_pwr_end = time.time()
                metrics_data = {
                    "total_energy_j": 0, "avg_gpu_w": 0, "avg_cpu_w": 0,
                    "measurement_mode": "Waiting_for_Host_WSL",
                    "gpu_energy_j": 0, "cpu_energy_j": 0,
                }
                total_energy_j = 0
                avg_power = 0
                idle_power = 0
            else:
                monitor = UnifiedPowerMonitor(OUTPUT_DIR, run_name, os_type, is_integrated=is_integrated)
                pwr_mode_str = "RAPL" if os_type == "linux" else "RAPL_Windows"
                print(f"  Power: {NUM_TEST_POWER} iterations (Using {pwr_mode_str})...")
                monitor.start()
                t_pwr_start = time.time()
                for i in range(NUM_TEST_POWER):
                    print(f"    Power run {i+1}/{NUM_TEST_POWER}...", end="\r", flush=True)
                    call_completion(port, multimodal_prompt, image_data, MAX_NEW_TOKENS, EFFECTIVE_TEMPERATURE)
                print()
                t_pwr_end = time.time()
                metrics_data = monitor.stop(global_start_time=t_pwr_start, global_end_time=t_pwr_end)
                total_energy_j = metrics_data["total_energy_j"]
                avg_power = metrics_data.get("avg_total_w", 0)
                idle_power = monitor.measure_baseline(duration_s=10.0)

            avg_power_adjusted = max(0, avg_power - idle_power) if not is_wsl_worker else avg_power

            ttft_stats = calculate_metrics(ttft_list) if ttft_list else {"mean": 0, "median": 0, "p95": 0}
            tpot_stats = calculate_metrics(tpot_list) if tpot_list else {"mean": 0, "median": 0, "p95": 0}
            energy_per_step_j = total_energy_j / max(1, NUM_TEST_POWER) if not is_wsl_worker else 0
            tokens_per_j = MAX_NEW_TOKENS / energy_per_step_j if energy_per_step_j > 0 else 0

            power_struct = {
                "avg_total_adjusted_w": avg_power_adjusted,
                "avg_idle_w": round(idle_power, 2) if not is_wsl_worker else 0,
                "energy_per_inference_j": energy_per_step_j,
                "tokens_per_joule_p50": tokens_per_j,
                "raw_energy_j": total_energy_j,
                "measurement_mode": metrics_data.get("measurement_mode", "unknown"),
            }
            if os_type == "linux" and not is_wsl_worker:
                power_struct["avg_gpu_w"] = metrics_data.get("avg_gpu_w", 0.0)
                power_struct["avg_cpu_w"] = metrics_data.get("avg_cpu_w", 0.0)
                power_struct["raw_gpu_energy_j"] = metrics_data.get("gpu_energy_j", 0.0)
                power_struct["raw_cpu_energy_j"] = metrics_data.get("cpu_energy_j", 0.0)

            res = {
                "config": run_name,
                "model": cfg["model_name"],
                "quant": quant,
                "backend": "llamacpp",
                "status": "ok",
                "is_integrated": is_integrated,
                "measurement_mode": metrics_data.get("measurement_mode", "unknown"),
                "latency": {
                    "ttft_ms_mean": ttft_stats["mean"], "ttft_ms_p50": ttft_stats["median"],
                    "ttft_ms_p95": ttft_stats["p95"],
                    "tpot_ms_mean": tpot_stats["mean"], "tpot_ms_p50": tpot_stats["median"],
                    "tpot_ms_p95": tpot_stats["p95"],
                },
                "power_energy": power_struct,
                "memory": {
                    "peak_mem_allocated_gb": peak_mem_allocated_gb,
                    "peak_mem_reserved_gb": peak_mem_reserved_gb,
                    **mem_details,
                },
                "bandwidth_analysis": {},
                "sanity": sanity,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if is_wsl_worker:
                res["wsl_pwr_start_epoch"] = t_pwr_start
                res["wsl_pwr_end_epoch"] = t_pwr_end
            if "hw_log_file" in metrics_data:
                res["hw_log_file"] = metrics_data["hw_log_file"]

            all_results.append(res)
            print(f"  Results: TTFT {ttft_stats['mean']:.0f}ms | TPOT {tpot_stats['mean']:.1f}ms/tok")

        except Exception as e:
            all_results.append({
                "config": run_name,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            })
            print(f"  Failed {run_name}: {type(e).__name__}: {e}")
        finally:
            stop_llama_server(server_proc)
            gc.collect()

    if is_wsl_worker:
        metrics_json = os.path.join(OUTPUT_DIR, "wsl_worker_metrics.json")
    else:
        metrics_json = get_timestamped_filename(OUTPUT_DIR, "benchmarker_metrics_llamacpp", "json")

    with open(metrics_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nFull metrics saved to: {metrics_json}")

    csv_path = os.path.join(OUTPUT_DIR, "benchmarker_metrics_llamacpp.csv")
    with open(csv_path, "w", newline="") as f:
        if all_results:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile SmolVLM GGUF models via llama-server")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file (required).")
    parser.add_argument("--wsl-worker", action="store_true", help="Internal flag. Do not use directly.")
    parser.add_argument("--wsl-output-dir", type=str, help="Output directory when in wsl-worker mode.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Config file not found: {args.config}")

    with open(args.config, "r") as f:
        user_config = json.load(f)

    config = dict_deep_update(DEFAULT_CONFIG.copy(), user_config)
    profile_llamacpp(config, args)