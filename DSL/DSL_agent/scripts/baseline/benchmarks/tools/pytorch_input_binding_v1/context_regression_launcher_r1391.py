from hecate_python_env import VENV,enter_nix
import shlex
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/"bin/python"),"-B","-m","unittest","application_execution.tests.test_context","test_application_artifact_adapter","test_application_model_source_context","test_pytorch_input_binding","test_program_output_tree_contract","-v"])
raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
