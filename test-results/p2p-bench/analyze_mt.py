#!/usr/bin/env python3
"""Analyze keyswitch_mt_bench nsys profiles.

nsys-rep is a SQLite DB. Measures (per thread):
  - CPU-side cudaLaunchKernel duration (RUNTIME events)
  - GPU-side kernel execution duration (KERNEL events)
  - ratio total-launch / total-compute

Warmup handling: bench runs 3 warmup rounds (33 kernel launches per thread)
then `rounds` measured rounds. We drop the first 40 launches per thread
(warmup + JIT outliers) and all kernels that start before the measure window.
"""
import sqlite3
import sys
import statistics

DB = sys.argv[1]
conn = sqlite3.connect(DB)
cur = conn.cursor()

# globalTid -> thread name
names = {}
for gid, nid in cur.execute("SELECT globalTid, nameId FROM ThreadNames"):
    row = cur.execute("SELECT value FROM StringIds WHERE id=?", (nid,)).fetchone()
    names[gid] = row[0] if row else f"tid{gid}"

# ---- launches (CPU side) ----
launch_rows = list(cur.execute(
    "SELECT globalTid, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME "
    "WHERE nameId = (SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%') "
    "ORDER BY globalTid, start"))

per_thread = {}
for tid, start, end in launch_rows:
    per_thread.setdefault(tid, []).append((start, end - start))

# drop first 40 per thread (3 warmup rounds x 11 kernels + margin), then JIT outliers
kept = {}
measure_start = {}
for tid, rows in per_thread.items():
    rows = rows[40:]
    rows = [r for r in rows if r[1] < 100_000_000]  # <100ms sane cap
    kept[tid] = rows
    # warmup threads do only 3 rounds (~177 launches); measure threads do
    # `rounds` rounds (~59 launches/round). Only measure threads set ms.
    if len(rows) > 1000:
        measure_start[tid] = rows[0][0]

print(f"=== {DB} ===")
print(f"{'thread':<28}{'n':>6}{'avg_us':>10}{'med_us':>10}{'p90_us':>10}{'min_us':>9}{'max_us':>9}{'total_ms':>11}")
tot_launch = 0.0
for tid in sorted(kept, key=lambda t: names.get(t, str(t))):
    rows = kept[tid]
    if len(rows) <= 1000:   # skip warmup-only threads (3 rounds)
        continue
    durs = [d for _, d in rows]
    tot = sum(durs) / 1e6
    tot_launch += tot
    print(f"{names.get(tid, tid):<28}{len(rows):>6}"
          f"{statistics.mean(durs)/1000:>10.2f}{statistics.median(durs)/1000:>10.2f}"
          f"{sorted(durs)[int(len(durs)*0.9)]/1000:>10.2f}"
          f"{min(durs)/1000:>9.2f}{max(durs)/1000:>9.2f}{tot:>11.3f}")

# ---- kernels (GPU side) ----
if measure_start:
    ms = min(measure_start.values())
kern_rows = list(cur.execute(
    "SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.shortName ORDER BY s.value, k.start"))
kern = {}
for name, start, end in kern_rows:
    if start < ms:
        continue
    kern.setdefault(name, []).append(end - start)

print()
print(f"{'kernel':<55}{'n':>6}{'avg_us':>10}{'med_us':>10}{'total_ms':>11}")
tot_compute = 0.0
for name in sorted(kern):
    durs = kern[name]
    tot = sum(durs) / 1e6
    tot_compute += tot
    print(f"{name:<55}{len(durs):>6}{statistics.mean(durs)/1000:>10.2f}"
          f"{statistics.median(durs)/1000:>10.2f}{tot:>11.3f}")

print(f"\nTOTAL launch (CPU)  = {tot_launch:.3f} ms")
print(f"TOTAL compute (GPU) = {tot_compute:.3f} ms")
print(f"launch/compute      = {tot_launch/tot_compute:.3f}")
conn.close()
