"""Attach opt-in native targets to an already configured benchmark GPU core."""
import argparse
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('cmake_source',type=Path)
args=parser.parse_args()
path=args.cmake_source/'CMakeLists.txt';text=path.read_text()
marker='if(DEFINED POSEIDON_IMAGENET_TARGETS_FILE)'
if marker not in text:
    path.write_text(text+'\n# Optional independent ImageNet native baselines.\n'+marker+
        '\n  include("${POSEIDON_IMAGENET_TARGETS_FILE}")\nendif()\n')
print('Opt-in native baseline targets configured; no default target behavior changed')
