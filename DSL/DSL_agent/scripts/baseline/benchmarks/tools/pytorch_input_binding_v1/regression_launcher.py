from hecate_python_env import VENV,enter_nix
import shlex
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/"bin/python"),"-B","-m","unittest",'test_pytorch_input_binding', 'test_pytorch_closed_public_arithmetic', 'test_pytorch_public_dtype_cast', 'test_pytorch_public_tensor_division', 'test_pytorch_public_boolean_arithmetic', 'test_pytorch_model_intake', 'test_pytorch_source_contract', 'test_application_model_source_context', 'test_program_output_tree_contract', 'test_application_component', 'test_application_model_source',"-v"])
raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
