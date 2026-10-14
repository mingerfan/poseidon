"""Multi-prime, batched N=65536 NTT, compiled by TileLang ahead of time.

Adapted from experiments/tile_lang_int32/tensorcore_ntt.py (TileLang 0.1.14).
Uses Poseidon's negacyclic DIF roots/TAM matrices and bit-reversed output.
All primes must be below 2^31. No Python is needed at benchmark runtime.
"""
import tilelang.language as T

N = 65536


@T.macro
def reduce64(value, q, ratio):
    # floor(2^64/q) gives a quotient at most one below the exact quotient.
    # For q < 2^31, the remaining value is < 2q < 2^32.
    remainder = T.cast(
        value - T.call_extern("uint64", "__umul64hi", value, ratio) * q,
        "uint32",
    )
    return T.if_then_else(remainder >= q, remainder - q, remainder)


def tensor_tam_stage(m: int):
    gap = N // (2 * m)
    rows = gap // 8
    assert rows >= 16 and rows % 16 == 0
    batch = T.dynamic("batch")
    limbs = T.dynamic("limbs")

    @T.prim_func
    def kernel(
        source: T.Tensor((batch, limbs, N), "uint32"),
        values: T.Tensor((batch, limbs, N), "uint32"),
        matrices: T.Tensor((limbs, m * 256), "uint32"),
        primes: T.Tensor((limbs,), "uint32"),
        ratios: T.Tensor((limbs,), "uint64"),
        weights: T.Tensor((limbs, 7), "uint32"),
    ):
        with T.Kernel(rows // 16, m, batch * limbs, threads=64) as (row_tile, group, plane):
            b = plane // limbs
            limb = plane % limbs
            q = primes[limb]
            ratio = ratios[limb]
            a_planes = T.alloc_shared((4, 16, 32), "uint8")
            b_planes = T.alloc_shared((4, 16, 32), "uint8")
            partial = T.alloc_fragment((16, 16), "int32")
            result = T.alloc_fragment((16, 16), "uint64")
            for k, row in T.Parallel(32, 16):
                if k < 16:
                    if m == 1:
                        a_value = source[b, limb, group * (2 * gap) + row_tile * 16 + row + k * rows]
                    else:
                        a_value = values[b, limb, group * (2 * gap) + row_tile * 16 + row + k * rows]
                    for a_byte in T.unroll(4):
                        a_planes[a_byte, row, k] = T.cast((a_value >> (8 * a_byte)) & 255, "uint8")
                else:
                    for a_byte in T.unroll(4):
                        a_planes[a_byte, row, k] = T.cast(0, "uint8")
            for col, k in T.Parallel(16, 32):
                if k < 16:
                    b_value = matrices[limb, group * 256 + col * 16 + k]
                    for b_byte in T.unroll(4):
                        b_planes[b_byte, col, k] = T.cast((b_value >> (8 * b_byte)) & 255, "uint8")
                else:
                    for b_byte in T.unroll(4):
                        b_planes[b_byte, col, k] = T.cast(0, "uint8")
            T.clear(result)
            for a_byte in T.unroll(4):
                for b_byte in T.unroll(4):
                    T.clear(partial)
                    T.gemm(a_planes[a_byte, :, :], b_planes[b_byte, :, :],
                           partial, transpose_B=True)
                    for row, col in T.Parallel(16, 16):
                        result[row, col] += T.cast(partial[row, col], "uint64") * T.cast(
                            weights[limb, a_byte + b_byte], "uint64")
            for row, col in T.Parallel(16, 16):
                values[b, limb, group * (2 * gap) + row_tile * 16 + row + col * rows] = reduce64(
                    result[row, col], q, ratio)

    return kernel.with_attr("global_symbol", f"encode_tile_tam_{m}")


def dif_fused2(m: int):
    """Two DIF stages per launch; each thread owns its quartet."""
    gap = N // (2 * m)
    assert gap >= 2
    batch = T.dynamic("batch")
    limbs = T.dynamic("limbs")

    @T.prim_func
    def kernel(
        source: T.Tensor((batch, limbs, N), "uint32"),
        values: T.Tensor((batch, limbs, N), "uint32"),
        roots: T.Tensor((limbs, N), "uint32"),
        primes: T.Tensor((limbs,), "uint32"),
        ratios: T.Tensor((limbs,), "uint64"),
    ):
        with T.Kernel(N // 1024, batch * limbs, threads=256) as (bx, plane):
            b = plane // limbs
            limb = plane % limbs
            q = primes[limb]
            ratio = ratios[limb]
            for t in T.Parallel(256):
                tid = bx * 256 + t
                group = tid // (gap // 2)
                j = tid % (gap // 2)
                base = group * (2 * gap) + j
                if m == 1:
                    a = source[b, limb, base]
                    v = source[b, limb, base + gap // 2]
                    c = source[b, limb, base + gap]
                    d = source[b, limb, base + gap + gap // 2]
                else:
                    a = values[b, limb, base]
                    v = values[b, limb, base + gap // 2]
                    c = values[b, limb, base + gap]
                    d = values[b, limb, base + gap + gap // 2]
                root1 = T.cast(roots[limb, m + group], "uint64")
                wc = reduce64(T.cast(c, "uint64") * root1, q, ratio)
                wd = reduce64(T.cast(d, "uint64") * root1, q, ratio)
                sum_ac = a + wc
                sum_bd = v + wd
                x0 = T.if_then_else(sum_ac >= q, sum_ac - q, sum_ac)
                x1 = T.if_then_else(sum_bd >= q, sum_bd - q, sum_bd)
                x2 = T.if_then_else(a >= wc, a - wc, a + q - wc)
                x3 = T.if_then_else(v >= wd, v - wd, v + q - wd)
                root2a = T.cast(roots[limb, 2 * m + 2 * group], "uint64")
                root2b = T.cast(roots[limb, 2 * m + 2 * group + 1], "uint64")
                w1 = reduce64(T.cast(x1, "uint64") * root2a, q, ratio)
                w3 = reduce64(T.cast(x3, "uint64") * root2b, q, ratio)
                sum01 = x0 + w1
                sum23 = x2 + w3
                values[b, limb, base] = T.if_then_else(sum01 >= q, sum01 - q, sum01)
                values[b, limb, base + gap // 2] = T.if_then_else(x0 >= w1, x0 - w1, x0 + q - w1)
                values[b, limb, base + gap] = T.if_then_else(sum23 >= q, sum23 - q, sum23)
                values[b, limb, base + gap + gap // 2] = T.if_then_else(x2 >= w3, x2 - w3, x2 + q - w3)

    return kernel.with_attr("global_symbol", f"encode_tile_dif_{m}")
