#!/usr/bin/env python3
"""
Analyze NSight Systems .nsys-rep / .sqlite files to extract:
1. Per-CUDA-API call timing
2. Per-NVTX-range: CUDA API calls and kernels grouped by handler operation
3. Kernel-to-NVTX correlation
4. Per-level breakdown (requires runtime compiled with level-infused NVTX names)

Usage: python analyze_nsys.py direct-4gpu.sqlite
       python analyze_nsys.py --level direct-4gpu.sqlite   # level-aware mode

NOTE on FHE "level":
  The current NVTX range names (e.g. "keyswitch.hybrid") do NOT embed the
  FHE modulus-chain level. To get per-level breakdowns, the runtime must be
  modified to include level info in NVTX range names (see below).
"""

import sqlite3
import sys
from collections import defaultdict


def analyze(db_path: str, level_mode: bool = False):
    _ = level_mode  # reserved for future use
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── 1. CUDA Runtime API calls ──────────────────────────────────────
    print("=" * 90)
    print("1. CUDA RUNTIME API CALLS — timing breakdown")
    print("=" * 90)
    rows = conn.execute("""
        SELECT s.value as api,
               COUNT(*) as calls,
               ROUND(AVG(r.end - r.start) / 1000.0, 1) as avg_us,
               ROUND(MIN(r.end - r.start) / 1000.0, 1) as min_us,
               ROUND(MAX(r.end - r.start) / 1000.0, 1) as max_us,
               ROUND(SUM(r.end - r.start) / 1000000.0, 3) as total_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY s.value
        ORDER BY total_ms DESC
    """).fetchall()

    print(f"{'CUDA API':<38s} {'calls':>7s} {'avg_us':>9s} {'min_us':>9s} {'max_us':>9s} {'total_ms':>10s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['api']:<38s} {r['calls']:>7d} {r['avg_us']:>9.1f} {r['min_us']:>9.1f} {r['max_us']:>9.1f} {r['total_ms']:>10.3f}")

    # ── 2. NVTX ranges — timing summary ───────────────────────────────
    print("\n" + "=" * 90)
    print("2. NVTX RANGES — GPU operation timing (no FHE level distinction)")
    print("=" * 90)
    nvtx = conn.execute("""
        SELECT text as nvtx_range,
               COUNT(*) as instances,
               ROUND(AVG(end - start) / 1000.0, 1) as avg_us,
               ROUND(SUM(end - start) / 1000000.0, 3) as total_ms
        FROM NVTX_EVENTS
        WHERE text IS NOT NULL AND end IS NOT NULL
        GROUP BY text
        ORDER BY total_ms DESC
    """).fetchall()

    print(f"{'NVTX Range':<55s} {'inst':>6s} {'avg_us':>9s} {'total_ms':>10s}")
    print("-" * 90)
    for r in nvtx:
        print(f"{r['nvtx_range']:<55s} {r['instances']:>6d} {r['avg_us']:>9.1f} {r['total_ms']:>10.3f}")

    # ── 3. Kernels grouped by NVTX range ──────────────────────────────
    print("\n" + "=" * 90)
    print("3. GPU KERNELS — grouped by NVTX operation")
    print("=" * 90)
    kernel_nvtx = conn.execute("""
        SELECT nt.text as nvtx_range,
               kt.shortName as kernel,
               COUNT(*) as launches,
               ROUND(AVG(k.end - k.start) / 1000.0, 1) as avg_us,
               ROUND(SUM(k.end - k.start) / 1000000.0, 3) as total_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN (SELECT id, value as shortName FROM StringIds) kt ON k.shortName = kt.id
        JOIN NVTX_EVENTS nt ON nt.text IS NOT NULL
             AND k.start >= nt.start AND (nt.end IS NULL OR k.start <= nt.end)
        GROUP BY nt.text, kt.shortName
        HAVING total_ms > 0.5
        ORDER BY nt.text, total_ms DESC
    """).fetchall()

    cur_nvtx = None
    for r in kernel_nvtx:
        if r['nvtx_range'] != cur_nvtx:
            cur_nvtx = r['nvtx_range']
            print(f"\n  [{cur_nvtx}]")
        print(f"    {r['kernel']:<62s} x{r['launches']:>5d} {r['avg_us']:>7.1f}us = {r['total_ms']:>8.3f}ms")

    # ── 4. CUDA API calls correlated with NVTX ranges ─────────────────
    print("\n" + "=" * 90)
    print("4. CUDA API — per NVTX operation (cudaLaunchKernel + cudaMemcpy*)")
    print("=" * 90)
    api_nvtx = conn.execute("""
        SELECT nt.text as nvtx_range,
               st.value as cuda_api,
               COUNT(*) as calls,
               ROUND(SUM(r.end - r.start) / 1000000.0, 3) as total_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds st ON r.nameId = st.id
        JOIN NVTX_EVENTS nt ON nt.text IS NOT NULL
             AND r.start >= nt.start AND (nt.end IS NULL OR r.start <= nt.end)
        WHERE st.value IN ('cudaLaunchKernel_ptsz_v7000', 'cudaMemcpyAsync_v3020',
                           'cudaMemcpy_v3020', 'cudaMemsetAsync_v3020')
        GROUP BY nt.text, st.value
        ORDER BY nt.text, total_ms DESC
    """).fetchall()

    cur_nvtx = None
    for r in api_nvtx:
        if r['nvtx_range'] != cur_nvtx:
            cur_nvtx = r['nvtx_range']
            print(f"\n  [{cur_nvtx}]")
        print(f"    {r['cuda_api']:<40s} x{r['calls']:>5d} = {r['total_ms']:>8.3f}ms")

    # ── 5. Per-device cudaLaunchKernel timing ─────────────────────────
    print("\n" + "=" * 90)
    print("5. cudaLaunchKernel — per device")
    print("=" * 90)
    dev_launch = conn.execute("""
        SELECT k.deviceId as dev,
               COUNT(*) as launches,
               ROUND(AVG(k.end - k.start) / 1000.0, 1) as avg_kernel_us,
               ROUND(SUM(k.end - k.start) / 1000000.0, 3) as total_kernel_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        GROUP BY k.deviceId
        ORDER BY k.deviceId
    """).fetchall()

    print(f"{'Device':>8s} {'kernel_launches':>17s} {'avg_us':>9s} {'total_ms':>12s}")
    print("-" * 50)
    for r in dev_launch:
        print(f"{'GPU '+str(r['dev']):>8s} {r['launches']:>17d} {r['avg_kernel_us']:>9.1f} {r['total_kernel_ms']:>12.3f}")

    # ── 6. LEVEL-AWARE analysis ────────────────────────────────────────
    # Works when NVTX names include " L{level}" suffix (after code change).
    # Strategy: for each kernel/API call, find the innermost NVTX range that
    # contains it AND has " L" in its name (level-labeled).
    has_levels = conn.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text LIKE '% L%'"
    ).fetchone()[0] > 0

    if has_levels:
        print("\n" + "=" * 90)
        print("6. PER-LEVEL: Kernel launches by FHE level")
        print("=" * 90)
        level_kernels = conn.execute("""
            WITH level_nvtx AS (
                SELECT start, end, text,
                       CAST(substr(text, instr(text, ' L') + 2) AS INTEGER) as level
                FROM NVTX_EVENTS
                WHERE text LIKE '% L%' AND end IS NOT NULL
            ),
            kernel_level AS (
                SELECT k.start as k_start, k.end as k_end, k.deviceId,
                       kt.shortName as kernel,
                       ln.level,
                       ln.text as nvtx_range,
                       ROW_NUMBER() OVER (
                           PARTITION BY k.start, k.end, k.deviceId
                           ORDER BY ln.end - ln.start ASC
                       ) as rn
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN (SELECT id, value as shortName FROM StringIds) kt
                     ON k.shortName = kt.id
                JOIN level_nvtx ln
                     ON k.start >= ln.start AND k.end <= ln.end
            )
            SELECT level, kernel,
                   COUNT(*) as launches,
                   ROUND(AVG(k_end - k_start) / 1000.0, 1) as avg_us,
                   ROUND(SUM(k_end - k_start) / 1000000.0, 3) as total_ms
            FROM kernel_level
            WHERE rn = 1
            GROUP BY level, kernel
            ORDER BY level DESC, total_ms DESC
        """).fetchall()

        cur_lvl = None
        for r in level_kernels:
            if r['level'] != cur_lvl:
                cur_lvl = r['level']
                print(f"\n  [Level {cur_lvl}]")
            print(f"    {r['kernel']:<62s} x{r['launches']:>5d} {r['avg_us']:>7.1f}us = {r['total_ms']:>8.3f}ms")

        print("\n" + "=" * 90)
        print("7. PER-LEVEL: CUDA API calls (cudaLaunchKernel + cudaMemcpy*)")
        print("=" * 90)
        level_api = conn.execute("""
            WITH level_nvtx AS (
                SELECT start, end, text,
                       CAST(substr(text, instr(text, ' L') + 2) AS INTEGER) as level
                FROM NVTX_EVENTS
                WHERE text LIKE '% L%' AND end IS NOT NULL
            ),
            api_level AS (
                SELECT r.start as r_start, r.end as r_end,
                       st.value as cuda_api,
                       ln.level,
                       ln.text as nvtx_range,
                       ROW_NUMBER() OVER (
                           PARTITION BY r.start, r.end
                           ORDER BY ln.end - ln.start ASC
                       ) as rn
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds st ON r.nameId = st.id
                JOIN level_nvtx ln
                     ON r.start >= ln.start AND r.end <= ln.end
                WHERE st.value IN ('cudaLaunchKernel_ptsz_v7000',
                                   'cudaMemcpyAsync_v3020',
                                   'cudaMemcpy_v3020',
                                   'cudaMemsetAsync_v3020')
            )
            SELECT level, cuda_api,
                   COUNT(*) as calls,
                   ROUND(AVG(r_end - r_start) / 1000.0, 1) as avg_us,
                   ROUND(SUM(r_end - r_start) / 1000000.0, 3) as total_ms
            FROM api_level
            WHERE rn = 1
            GROUP BY level, cuda_api
            ORDER BY level DESC, total_ms DESC
        """).fetchall()

        cur_lvl = None
        for r in level_api:
            if r['level'] != cur_lvl:
                cur_lvl = r['level']
                print(f"\n  [Level {cur_lvl}]")
            print(f"    {r['cuda_api']:<40s} x{r['calls']:>5d} avg={r['avg_us']:>7.1f}us total={r['total_ms']:>8.3f}ms")

        print("\n" + "=" * 90)
        print("8. PER-LEVEL: Same kernel name across different levels")
        print("=" * 90)
        same_kernel = conn.execute("""
            WITH level_nvtx AS (
                SELECT start, end, text,
                       CAST(substr(text, instr(text, ' L') + 2) AS INTEGER) as level
                FROM NVTX_EVENTS
                WHERE text LIKE '% L%' AND end IS NOT NULL
            ),
            kernel_level AS (
                SELECT k.start as k_start, k.end as k_end,
                       kt.shortName as kernel,
                       ln.level,
                       ROW_NUMBER() OVER (
                           PARTITION BY k.start, k.end
                           ORDER BY ln.end - ln.start ASC
                       ) as rn
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN (SELECT id, value as shortName FROM StringIds) kt
                     ON k.shortName = kt.id
                JOIN level_nvtx ln
                     ON k.start >= ln.start AND k.end <= ln.end
            )
            SELECT kernel, level,
                   COUNT(*) as launches,
                   ROUND(AVG(k_end - k_start) / 1000.0, 1) as avg_us
            FROM kernel_level
            WHERE rn = 1
            GROUP BY kernel, level
            HAVING COUNT(DISTINCT level) >= 3
            ORDER BY kernel, level DESC
        """).fetchall()

        cur_kernel = None
        for r in same_kernel:
            if r['kernel'] != cur_kernel:
                cur_kernel = r['kernel']
                print(f"\n  [{cur_kernel}]")
            print(f"    L{r['level']:<5d} x{r['launches']:>5d} avg={r['avg_us']:>6.1f}us")
    else:
        print("\n" + "=" * 90)
        print("NOTE: No level-infused NVTX data found.")
        print("Rebuild with the modified gpu_keyswitch_handler.cpp /")
        print("gpu_modswitch_handler.cpp and re-profile to enable sections 6-8.")
        print("=" * 90)

    conn.close()


def main():
    level_mode = False
    args = sys.argv[1:]
    if '--level' in args:
        level_mode = True
        args.remove('--level')
    if len(args) < 1:
        print(f"Usage: {sys.argv[0]} [--level] <path-to-.sqlite-or-.nsys-rep>")
        sys.exit(1)
    analyze(args[0], level_mode)


if __name__ == "__main__":
    main()
