# -*- coding: utf-8 -*-
"""
PhysioNet 2019 Sepsis Challenge Dataset Ingestion Tool
Official Dataset: Early Prediction of Sepsis from Clinical Data (PhysioNet/CinC Challenge 2019)
Source Base URL: https://physionet.org/files/challenge-2019/1.0.0/training/
"""

import os
import sys
import urllib.request
import urllib.error
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL_A = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/"
BASE_URL_B = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/"

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

def fetch_patient_file(args):
    url, output_path = args
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True, "cached"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            with open(output_path, 'wb') as f:
                f.write(content)
        return True, "downloaded"
    except Exception as e:
        return False, str(e)

def download_dataset(num_set_a=1500, num_set_b=1000, max_workers=20):
    """
    Downloads PhysioNet 2019 patient PSV files into data/raw/
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    set_a_dir = os.path.join(RAW_DIR, "setA")
    set_b_dir = os.path.join(RAW_DIR, "setB")
    os.makedirs(set_a_dir, exist_ok=True)
    os.makedirs(set_b_dir, exist_ok=True)

    tasks = []
    
    # Queue Set A tasks (p000001 to p000001+num_set_a)
    for i in range(1, num_set_a + 1):
        pid_str = f"p{i:06d}"
        url = f"{BASE_URL_A}{pid_str}.psv"
        out_path = os.path.join(set_a_dir, f"{pid_str}.psv")
        tasks.append((url, out_path))

    # Queue Set B tasks (p100001 to p100001+num_set_b)
    for i in range(100001, 100001 + num_set_b):
        pid_str = f"p{i:06d}"
        url = f"{BASE_URL_B}{pid_str}.psv"
        out_path = os.path.join(set_b_dir, f"{pid_str}.psv")
        tasks.append((url, out_path))

    print(f"[*] Fetching {len(tasks)} clinical patient records from PhysioNet 2019...")
    start_time = time.time()
    
    success_count = 0
    cached_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_patient_file, t): t for t in tasks}
        for future in as_completed(futures):
            ok, status = future.result()
            if ok:
                if status == "cached":
                    cached_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1

    elapsed = time.time() - start_time
    print(f"[OK] Download finished in {elapsed:.2f}s!")
    print(f"   Downloaded: {success_count} files | Cached: {cached_count} files | Failed: {fail_count} files")
    return success_count + cached_count

if __name__ == "__main__":
    download_dataset()
