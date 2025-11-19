import multiprocessing
multiprocessing.freeze_support()

import time
import json
import subprocess
import os
from pathlib import Path
import requests
from multiprocessing import Pool
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

###############################################################################
# WINDOWS: ABSOLUTE PATH TO GCLOUD
###############################################################################

GCLOUD_PATH = r"C:\Users\shubh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

###############################################################################
# LOAD ENV VARIABLES
###############################################################################

load_dotenv()

ENDPOINT_URL = os.environ.get("ENDPOINT_URL")
PROJECT_ID   = os.environ.get("PROJECT_ID")
LOCATION     = os.environ.get("LOCATION")
ENDPOINT_ID  = os.environ.get("ENDPOINT_ID")
MODEL_ID     = os.environ.get("MODEL_ID")

print("CONFIGURATION:")
print("  ENDPOINT_URL:", ENDPOINT_URL)
print("  PROJECT_ID:  ", PROJECT_ID)
print("  LOCATION:    ", LOCATION)
print("  ENDPOINT_ID: ", ENDPOINT_ID)
print("  MODEL_ID:    ", MODEL_ID)


###############################################################################
# LOAD INPUT FILE (JSONL)
###############################################################################

INPUT_FILE = "combined_benchmarks_648.jsonl"

data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except Exception as e:
            print("⚠️ Skipping malformed input:", e)

print(f"Loaded {len(data)} records from {INPUT_FILE}")


###############################################################################
# CONFIGURATION
###############################################################################

WORKERS = 2
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5
MAX_RUNTIME_HOURS = 4.5
TOKEN_REFRESH_INTERVAL = 45 * 60  # refresh every 45 min

REQUEST_TIMEOUT = 600  # <-- NEW 5 MIN TIMEOUT

OUT_DIR = Path("Milestone-5/math-agent")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUTPUT = OUT_DIR / f"{ENDPOINT_URL}_predictions_merged.jsonl"


###############################################################################
# TOKEN REFRESH (GCLOUD)
###############################################################################

def refresh_access_token():
    try:
        result = subprocess.run(
            [GCLOUD_PATH, "auth", "print-access-token"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("❌ Token refresh failed:", result.stderr.strip())
            return None
        return result.stdout.strip()

    except Exception as e:
        print("❌ Token refresh exception:", e)
        return None


###############################################################################
# LOAD PROCESSED INDICES (CHECKPOINT)
###############################################################################

def load_processed_indices(path):
    processed = set()
    if not path.exists():
        return processed
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "index" in obj:
                        processed.add(obj["index"])
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
    return processed


###############################################################################
# WORKER
###############################################################################

def run_worker(args):
    worker_id, records, global_start = args

    out_file = OUT_DIR / f"{ENDPOINT_URL}_worker_{worker_id}.jsonl"
    print(f"[Worker {worker_id}] Starting → {out_file}")

    processed_indices = load_processed_indices(out_file)
    if processed_indices:
        print(f"[Worker {worker_id}] Resuming ({len(processed_indices)} done).")

    session = requests.Session()
    ACCESS_TOKEN = refresh_access_token()
    if not ACCESS_TOKEN:
        print(f"[Worker {worker_id}] ❌ Could not obtain initial token.")
        return

    last_refresh = time.time()
    job_start = time.time()
    max_runtime = MAX_RUNTIME_HOURS * 3600

    try:
        with out_file.open("a", encoding="utf-8") as outf:

            for local_idx, record in enumerate(records):
                idx = global_start + local_idx

                if idx in processed_indices:
                    continue

                # Timeout check
                if time.time() - job_start > max_runtime:
                    print(f"[Worker {worker_id}] ⛔ Time limit reached.")
                    return

                # Extract prompt
                messages = record.get("body", {}).get("messages", [])
                temperature = record.get("temperature", 0)

                prompt = None
                for m in reversed(messages):
                    if m.get("role") == "user":
                        prompt = m.get("content")
                        break

                if not prompt:
                    processed_indices.add(idx)
                    continue

                payload = {
                    "model": MODEL_ID,
                    "temperature": temperature,
                    "messages": messages,
                }

                # PROACTIVE TOKEN REFRESH
                if time.time() - last_refresh > TOKEN_REFRESH_INTERVAL:
                    print(f"[Worker {worker_id}] 🔄 Proactive token refresh...")
                    new_token = refresh_access_token()
                    if not new_token:
                        print(f"[Worker {worker_id}] ❌ Refresh failed.")
                        return
                    ACCESS_TOKEN = new_token
                    last_refresh = time.time()

                # ---------------- RETRY LOOP ----------------
                attempt = 0
                while attempt < MAX_RETRIES:
                    try:
                        url = (
                            f"https://{ENDPOINT_URL}/v1beta1/projects/{PROJECT_ID}/"
                            f"locations/{LOCATION}/endpoints/{ENDPOINT_ID}/chat/completions"
                        )

                        resp = session.post(
                            url,
                            headers={
                                "Authorization": f"Bearer {ACCESS_TOKEN}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                            timeout=REQUEST_TIMEOUT,  # <-- 5 minutes
                        )

                        try:
                            body = resp.json()
                        except:
                            body = {"raw": resp.text}

                        # TOKEN EXPIRED
                        if resp.status_code == 401:
                            print(f"[Worker {worker_id}] ⚠️ 401 at idx {idx}. Refreshing…")
                            new_token = refresh_access_token()
                            if not new_token:
                                outf.write(json.dumps({"index": idx, "error": "token refresh failed"})+"\n")
                                outf.flush()
                                break
                            ACCESS_TOKEN = new_token
                            last_refresh = time.time()
                            attempt += 1
                            time.sleep((2 ** attempt) * 2)  # <-- backoff improved
                            continue

                        # API ERROR (non-200)
                        if resp.status_code != 200:
                            outf.write(json.dumps({"index": idx, "error": body})+"\n")
                            outf.flush()
                            print(f"[Worker {worker_id}] ❌ API error {resp.status_code} at idx {idx}. Skipping.")
                            processed_indices.add(idx)
                            break

                        # Payload error
                        if isinstance(body, dict) and "error" in body:
                            outf.write(json.dumps({"index": idx, "error": body})+"\n")
                            outf.flush()
                            processed_indices.add(idx)
                            print(f"[Worker {worker_id}] ❌ Payload error at idx {idx}.")
                            break

                        # SUCCESS
                        result = {
                            "index": idx,
                            "status": resp.status_code,
                            "prompt": prompt,
                            "response": body,
                        }
                        outf.write(json.dumps(result)+"\n")
                        outf.flush()

                        processed_indices.add(idx)
                        print(f"[Worker {worker_id}] ✓ {idx} processed")

                        time.sleep(SLEEP_BETWEEN_CALLS)
                        break  # exit retry loop

                    except Exception as e:
                        attempt += 1
                        print(f"[Worker {worker_id}] ⚠ Exception at idx {idx}: {e}. Retry {attempt}/{MAX_RETRIES}")

                        if attempt >= MAX_RETRIES:
                            outf.write(json.dumps({
                                "index": idx,
                                "error": f"timeout after {MAX_RETRIES} retries: {str(e)}"
                            })+"\n")
                            outf.flush()
                            processed_indices.add(idx)
                            print(f"[Worker {worker_id}] ⚠ Max retries reached at idx {idx}. Skipping.")
                            break

                        time.sleep((2 ** attempt) * 2)  # <-- exponential backoff: 2,4,8s

    finally:
        session.close()
        print(f"[Worker {worker_id}] Finished.")


###############################################################################
# PARALLEL MERGING
###############################################################################

def read_worker_file(path):
    recs = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "index" in obj:
                        recs.append(obj)
                except:
                    continue
    except:
        pass
    return recs


def parallel_merge_worker_outputs(worker_count, merged_path):
    worker_files = [OUT_DIR / f"{ENDPOINT_URL}_worker_{wid}.jsonl" for wid in range(worker_count)]
    all_records = []

    with ThreadPoolExecutor(max_workers=min(8, worker_count)) as exe:
        futures = {exe.submit(read_worker_file, p): p for p in worker_files}
        for fut in as_completed(futures):
            recs = fut.result()
            all_records.extend(recs)
            print(f"Read {len(recs)} from {futures[fut].name}")

    all_records.sort(key=lambda x: x["index"])

    with merged_path.open("w", encoding="utf-8") as mf:
        for obj in all_records:
            mf.write(json.dumps(obj)+"\n")

    print(f"Merged {len(all_records)} entries → {merged_path}")


###############################################################################
# MAIN DRIVER
###############################################################################

def make_jobs(total, workers):
    chunk = (total + workers - 1) // workers
    jobs = []
    for wid in range(workers):
        start = wid * chunk
        end = min(start + chunk, total)
        if start >= total:
            break
        jobs.append((wid, data[start:end], start))
    return jobs


if __name__ == "__main__":
    total = len(data)
    jobs = make_jobs(total, WORKERS)

    print(f"🚀 Launching {len(jobs)} workers for {total} datapoints...")

    pool = Pool(processes=len(jobs))

    try:
        pool.map(run_worker, jobs)
    except KeyboardInterrupt:
        print("⚠️ Interrupted — terminating workers...")
        pool.terminate()
    finally:
        pool.join()

    print("🏁 Workers finished.")
    print("🔄 Merging results...")

    parallel_merge_worker_outputs(len(jobs), MERGED_OUTPUT)

    print("\n✨ DONE!")
    print(f"Merged output saved in:\n  {MERGED_OUTPUT}")
