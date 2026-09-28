"""Lossless CST serialization bridge for the fixed 16384-slot SEAL runtime.

SEAL_HEVM.cpp encode_internal repeats src[i % src.size()]. A full vector can
therefore be shortened only when every period block is byte-identical. This
adapter does not change helper code, the encoded slot sequence, or gate policy.
"""
import hashlib
import math
import struct

SLOTS = 16384

def canonicalize(data, period):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(type(period) is int and period in (4,8,16,32,64,128,256), "Invalid CST period")
    require(type(data) is bytes and 8 <= len(data) <= 1024**2, "Invalid CST size")
    count, = struct.unpack_from("<q", data)
    require(0 <= count <= 256, "Invalid constant count")
    offset = 8
    output = [data[:8]]
    changes = []
    for index in range(count):
        require(offset+8 <= len(data), "Truncated CST length")
        size, = struct.unpack_from("<q", data, offset)
        offset += 8
        require(size in (1, period, SLOTS) and offset+size*8 <= len(data), "Invalid CST vector size")
        raw = data[offset:offset+size*8]
        offset += len(raw)
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", raw)), "Non-finite constant")
        if size == SLOTS:
            block = raw[:period*8]
            require(raw == block*(SLOTS//period), "Non-periodic full CST vector")
            raw = block
            changes.append(dict(index=index, original_size=size, canonical_size=period))
            size = period
        output.extend((struct.pack("<q", size), raw))
    require(offset == len(data), "Trailing CST data")
    canonical = b"".join(output)
    record = dict(schema=1, method="exact-byte-period-repetition-v1", period=period,
                  slot_count=SLOTS, changes=changes,
                  original_sha256=hashlib.sha256(data).hexdigest(),
                  canonical_sha256=hashlib.sha256(canonical).hexdigest())
    return canonical, record
