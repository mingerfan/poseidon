"""Generate the fixed synthetic weights and two-token input; NumPy only."""
import argparse
import json
from qwen2token.fixture import prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    record=prepare(args.output_dir)
    print(json.dumps({k:v for k,v in record.items() if k!='weight_arrays'},indent=2))
