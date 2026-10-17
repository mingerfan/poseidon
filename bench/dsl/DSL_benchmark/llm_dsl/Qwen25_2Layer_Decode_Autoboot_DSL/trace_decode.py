"""Trace the frozen two-block, two-token decode DSL using native Hecate."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    entry = Path(__file__).resolve().parent / 'source/Qwen25_2Layer_Decode_Test/model_entry.py'
    runpy.run_path(str(entry), run_name='__main__')
