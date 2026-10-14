"""Manual native object-array candidates through real sandbox/SEAL, never API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES = ["matrix_reverse","matrix_transpose","rank_four","row_unpack","mixed_plain","zero_item","zero_index","zero_return","empty_array","nested_array","wrong_transpose"]

if __name__ == '__main__':
    raise SystemExit(main(
        plans=[('arithmetic-alias-chain',name,name.startswith('wrong_')) for name in CASES],
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_arrays',
        prefix='native-array-goldens-',title='Native object-array goldens',input_counts=(1,),
        extra_options=('--native-arrays',)))
