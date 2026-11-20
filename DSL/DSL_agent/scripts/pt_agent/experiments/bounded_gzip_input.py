"""Prepare one gzip member as input using raw DEFLATE, without footer checks.

This bounded input preparation parses the standard header and streams decoded
bytes into a fresh destination. CRC, ISIZE, FHCRC, hashes and decoded-size
identity comparisons are deliberately absent. The caller owns resource guards.
"""
import os
from pathlib import Path
import zlib

CHUNK = 1024 * 1024


def restore_raw_member(source, target, *, maximum_output_bytes, guard):
    """Write one fresh raw input, retaining partial output on any failure."""
    source, target = Path(source), Path(target)
    if type(maximum_output_bytes) is not int or maximum_output_bytes <= 0:
        raise ValueError('A finite positive output-file allowance is required')
    if os.path.lexists(target):
        raise ValueError('Preserve an existing restored input or partial input')
    read_bytes = 0
    decoded_bytes = 0
    input_blocks = 0
    output_blocks = 0
    guard(0)
    with source.open('rb') as reader, target.open('xb') as sink:
        def take(amount):
            nonlocal read_bytes
            result = bytearray()
            while len(result) < amount:
                guard(0)
                part = reader.read(min(CHUNK, amount - len(result)))
                if not part:
                    raise EOFError('Truncated gzip header or optional header field')
                result.extend(part)
                read_bytes += len(part)
            return bytes(result)

        header = take(10)
        if header[:3] != b'\x1f\x8b\x08' or header[3] & 0xe0:
            raise ValueError('Unsupported gzip header format')
        flags = header[3]
        if flags & 0x04:
            take(int.from_bytes(take(2), 'little'))
        for flag in (0x08, 0x10):
            if flags & flag:
                while take(1) != b'\x00':
                    pass
        if flags & 0x02:
            take(2)  # FHCRC bytes are skipped without interpretation or comparison.
        decoder = zlib.decompressobj(wbits=-15)
        payload = b''
        while not decoder.eof:
            guard(0)
            if not payload:
                payload = reader.read(CHUNK)
                if not payload:
                    raise EOFError('Truncated raw DEFLATE stream before its end marker')
                read_bytes += len(payload)
                input_blocks += 1
            previous = len(payload)
            try:
                decoded = decoder.decompress(payload, CHUNK)
            except zlib.error as exc:
                raise ValueError('Invalid raw DEFLATE stream: ' + str(exc)) from exc
            payload = decoder.unconsumed_tail
            if decoded:
                if decoded_bytes + len(decoded) > maximum_output_bytes:
                    raise RuntimeError('resource_limit: restored input exceeds its output-file allowance')
                guard(len(decoded))
                sink.write(decoded)
                decoded_bytes += len(decoded)
                output_blocks += 1
            elif not decoder.eof and len(payload) == previous:
                raise ValueError('Raw DEFLATE decoder made no progress')
        # Stop at raw DEFLATE EOF. The gzip footer is neither parsed nor checked.
        sink.flush()
        os.fsync(sink.fileno())
    guard(0)
    return dict(status='ok', source=str(source), restored_input=str(target),
        compressed_bytes_read=read_bytes, decoded_bytes_written=decoded_bytes,
        compressed_blocks=input_blocks, decoded_blocks=output_blocks, block_bytes=CHUNK,
        footer_parsed=False, CRC_checks=0, ISIZE_checks=0, FHCRC_checks=0,
        hash_checks=0, decoded_size_identity_comparisons=0,
        scope='Bounded fresh-case input preparation; raw DEFLATE EOF and format only; no extra integrity verification')
