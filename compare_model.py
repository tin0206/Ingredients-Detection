import pandas as pd
from ultralytics import YOLO, RTDETR
import os
import psutil
import GPUtil
import torch
import time
import gc
import threading

from concurrent.futures import ThreadPoolExecutor


# =========================================================
# RESET CUDA / SYSTEM
# =========================================================
def reset_system_and_cuda():
    """Giải phóng bộ nhớ trước mỗi benchmark"""

    gc.collect()

    if torch.cuda.is_available():

        try:
            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

            torch.cuda.synchronize()

        except Exception as e:
            print(f"CUDA reset warning: {e}")

    time.sleep(1)


# =========================================================
# HARDWARE STATUS
# =========================================================
def get_hardware_status():

    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)

    gpu_load = 0
    vram_used_mb = 0

    if torch.cuda.is_available():

        gpus = GPUtil.getGPUs()

        if gpus:
            gpu_load = gpus[0].load * 100

        vram_used_mb = (
            torch.cuda.memory_allocated(0) / (1024 ** 2)
        )

    return (
        round(ram_used_gb, 2),
        round(gpu_load, 1),
        round(vram_used_mb, 1)
    )


# =========================================================
# GPU UTILIZATION MONITOR
# =========================================================
class GPUMonitor:
    """
    Theo dõi peak GPU utilization realtime
    """

    def __init__(self):

        self.running = False

        self.peak_gpu = 0

    def monitor(self):

        while self.running:

            try:

                gpus = GPUtil.getGPUs()

                if gpus:

                    current_load = gpus[0].load * 100

                    if current_load > self.peak_gpu:
                        self.peak_gpu = current_load

            except:
                pass

            time.sleep(0.05)

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.monitor
        )

        self.thread.daemon = True

        self.thread.start()

    def stop(self):

        self.running = False

        self.thread.join()


# =========================================================
# MAIN BENCHMARK
# =========================================================
def run_evaluation_and_stress():

    # =====================================================
    # CONFIG
    # =====================================================
    model_configs = {

        "YOLO26s_150epochs": {
            "path": "runs/detect/train-4/weights/best.pt",
            "imgsz": 640
        },

        "RTDETR_L_150epochs": {
            "path": "runs/detect/train/weights/best.pt",
            "imgsz": 640
        }
    }

    data_yaml = "data.yaml"

    tests = [
        "test.jpg",
        "test2.jpg",
        "test3.jpg",
        "test4.jpg",
        "test5.jpg",
        "test6.jpg",
        "test7.jpg",
        "test8.jpg"
    ]

    valid_tests = [
        img for img in tests
        if os.path.exists(img)
    ]

    if not valid_tests:
        print("⚠️ No test images found!")

    stress_milestones = [

        {"total_imgs": 10, "concurrent_users": 2},

        {"total_imgs": 50, "concurrent_users": 5},

        {"total_imgs": 100, "concurrent_users": 10},

        {"total_imgs": 500, "concurrent_users": 25},

        {"total_imgs": 1000, "concurrent_users": 50}
    ]

    overall_results = []

    stress_results = []

    # =====================================================
    # MAIN LOOP
    # =====================================================
    for name, config in model_configs.items():

        if not os.path.exists(config["path"]):

            print(f"⚠️ Missing weights: {config['path']}")

            continue

        print("\n====================================================")
        print(f"PROCESSING MODEL: {name}")
        print("====================================================")

        # =================================================
        # RESET
        # =================================================
        reset_system_and_cuda()

        # =================================================
        # LOAD MODEL
        # =================================================
        model = (
            RTDETR(config["path"])
            if "RTDETR" in name
            else YOLO(config["path"])
        )

        # =================================================
        # PART 1 - TEST SET EVALUATION
        # =================================================
        print("\n--- 1. Test Set Evaluation ---")

        metrics = model.val(
            data=data_yaml,
            split="test",
            imgsz=config["imgsz"],
            batch=16,
            conf=0.1,
            iou=0.6,
            verbose=False
        )

        ram_used, _ , vram_used = get_hardware_status()

        if torch.cuda.is_available():

            vram_used = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        preprocess_time = metrics.speed.get(
            "preprocess", 0
        )

        inference_time = metrics.speed.get(
            "inference", 0
        )

        postprocess_time = metrics.speed.get(
            "postprocess", 0
        )

        total_latency = (
            preprocess_time +
            inference_time +
            postprocess_time
        )

        # =================================================
        # PART 2 - SINGLE STREAM BENCHMARK
        # =================================================
        print("\n--- 2. Single Stream Benchmark ---")

        single_stream_fps = 0

        single_stream_latency = 0

        if valid_tests:

            benchmark_imgs = (
                valid_tests *
                (100 // len(valid_tests) + 1)
            )[:100]

            reset_system_and_cuda()

            # Warmup
            model(
                benchmark_imgs[:2],
                imgsz=config["imgsz"],
                conf=0.1,
                verbose=False
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start_single = time.time()

            for img in benchmark_imgs:

                model(
                    img,
                    imgsz=config["imgsz"],
                    conf=0.1,
                    verbose=False
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            end_single = time.time()

            single_duration = (
                end_single - start_single
            )

            single_stream_fps = (
                len(benchmark_imgs)
                / single_duration
            )

            single_stream_latency = (
                single_duration
                / len(benchmark_imgs)
            ) * 1000

            print(
                f"Single Stream FPS: "
                f"{round(single_stream_fps, 1)} | "
                f"Latency/Image: "
                f"{round(single_stream_latency, 2)} ms"
            )

        # =================================================
        # SAVE OVERALL RESULTS
        # =================================================
        file_size_mb = (
            os.path.getsize(config["path"])
            / (1024 * 1024)
        )

        num_params = sum(
            p.numel()
            for p in model.model.parameters()
        )

        overall_results.append({

            "Model": name,

            "Size_MB": round(file_size_mb, 1),

            "Params_M": round(num_params / 1e6, 2),

            "Val_RAM_GB": ram_used,

            "Val_VRAM_Peak_MB": round(
                vram_used, 1
            ),

            "Precision": round(
                metrics.results_dict.get(
                    "metrics/precision(B)", 0
                ),
                4
            ),

            "Recall": round(
                metrics.results_dict.get(
                    "metrics/recall(B)", 0
                ),
                4
            ),

            "mAP50": round(
                metrics.results_dict.get(
                    "metrics/mAP50(B)", 0
                ),
                4
            ),

            "mAP50-95": round(
                metrics.results_dict.get(
                    "metrics/mAP50-95(B)", 0
                ),
                4
            ),

            "Inference_ms": round(
                inference_time, 2
            ),

            "Total_Latency_ms": round(
                total_latency, 2
            ),

            "Val_FPS": round(
                1000 / total_latency,
                1
            ) if total_latency > 0 else 0,

            "Single_Stream_FPS": round(
                single_stream_fps,
                1
            ),

            "Single_Stream_Latency_ms": round(
                single_stream_latency,
                2
            )
        })

        # =================================================
        # PART 3 - STRESS TEST
        # =================================================
        print("\n--- 3. Concurrent Stress Test ---")

        for milestone in stress_milestones:

            total_imgs = milestone["total_imgs"]

            users = milestone["concurrent_users"]

            reset_system_and_cuda()

            simulated_batch = (
                valid_tests *
                (total_imgs // len(valid_tests) + 1)
            )[:total_imgs]
            
            if total_imgs < 20:
                print("Skipping tiny workload for stable benchmark")
                continue

            print(
                f"\n-> Simulating "
                f"{users} users "
                f"with {total_imgs} images"
            )

            # =============================================
            # WARMUP
            # =============================================
            try:

                model(
                    simulated_batch[:2],
                    imgsz=config["imgsz"],
                    conf=0.1,
                    verbose=False
                )

            except:
                pass

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # =============================================
            # SPLIT WORKLOAD
            # =============================================
            chunk_size = max(
                1,
                total_imgs // users
            )

            user_batches = [

                simulated_batch[i:i + chunk_size]

                for i in range(
                    0,
                    total_imgs,
                    chunk_size
                )
            ]

            def run_user_batch(batch_imgs):

                model(
                    batch_imgs,
                    imgsz=config["imgsz"],
                    conf=0.1,
                    verbose=False,
                    stream=False
                )

            # =============================================
            # GPU MONITOR
            # =============================================
            gpu_monitor = GPUMonitor()

            gpu_monitor.start()

            # =============================================
            # BENCHMARK START
            # =============================================
            repeat_runs = 3

            all_durations = []

            for _ in range(repeat_runs):

                torch.cuda.synchronize() if torch.cuda.is_available() else None

                start_time = time.time()

                with ThreadPoolExecutor(
                    max_workers=users
                ) as executor:

                    list(executor.map(
                        run_user_batch,
                        user_batches
                    ))

                torch.cuda.synchronize() if torch.cuda.is_available() else None

                end_time = time.time()

                all_durations.append(
                    end_time - start_time
                )

            # =============================================
            # GPU MONITOR STOP
            # =============================================
            gpu_monitor.stop()

            # Average duration
            duration = (
                sum(all_durations)
            )

            # =============================================
            # METRICS
            # =============================================
            stress_fps = (
                total_imgs / duration
            )

            avg_time_per_user = (
                duration / users
            )

            avg_time_per_image = (
                duration / total_imgs
            )

            stress_ram, _, _ = (
                get_hardware_status()
            )

            # FIXED GPU UTIL
            stress_gpu_load = round(
                gpu_monitor.peak_gpu,
                1
            )

            if torch.cuda.is_available():

                stress_vram_peak = (
                    torch.cuda.max_memory_allocated()
                    / (1024 ** 2)
                )

            else:
                stress_vram_peak = 0

            stress_results.append({

                "Model": name,

                "Image_Count": total_imgs,

                "Concurrent_Users": users,

                "Total_Time_Sec": round(
                    duration, 2
                ),

                "Avg_FPS": round(
                    stress_fps, 1
                ),

                "Avg_Time_Per_User_Sec": round(
                    avg_time_per_user,
                    2
                ),

                "Avg_Time_Per_Image_Sec": round(
                    avg_time_per_image,
                    4
                ),

                "Peak_VRAM_MB": round(
                    stress_vram_peak,
                    1
                ),

                "RAM_GB": stress_ram,

                "GPU_Load_Percent": stress_gpu_load
            })

            print(
                f"Finished: "
                f"{round(duration, 2)}s | "
                f"FPS: {round(stress_fps, 1)} | "
                f"Avg/User: "
                f"{round(avg_time_per_user, 2)}s | "
                f"Avg/Image: "
                f"{round(avg_time_per_image, 4)}s | "
                f"GPU Load: "
                f"{stress_gpu_load}%"
            )

        # =================================================
        # CLEANUP
        # =================================================
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================
    # EXPORT CSV
    # =====================================================
    print("\n====================================================")
    print("EXPORTING REPORTS")
    print("====================================================")

    # =====================================================
    # OVERALL RESULTS
    # =====================================================
    if overall_results:

        df_overall = pd.DataFrame(
            overall_results
        )

        df_overall.to_csv(
            "test_set_comparison_results_final.csv",
            index=False
        )

        print("\n=== OVERALL RESULTS ===")

        print(df_overall[[

            "Model",

            "mAP50",

            "mAP50-95",

            "Inference_ms",

            "Val_FPS",

            "Single_Stream_FPS",

            "Single_Stream_Latency_ms",

            "Val_VRAM_Peak_MB"

        ]].to_string(index=False))

    # =====================================================
    # STRESS RESULTS
    # =====================================================
    if stress_results:

        df_stress = pd.DataFrame(
            stress_results
        )

        df_stress.to_csv(
            "hardware_stress_test_results.csv",
            index=False
        )

        print("\n=== STRESS TEST RESULTS ===")

        print(df_stress[[

            "Model",

            "Image_Count",

            "Concurrent_Users",

            "Total_Time_Sec",

            "Avg_FPS",

            "Avg_Time_Per_User_Sec",

            "Avg_Time_Per_Image_Sec",

            "Peak_VRAM_MB",

            "GPU_Load_Percent"

        ]].to_string(index=False))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    run_evaluation_and_stress()