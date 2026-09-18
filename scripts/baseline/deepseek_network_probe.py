"""No-key GET probe in the same pinned/isolated Python as the Agent transport.

Does not invoke model generation, read credentials or change system settings.
"""
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    if "--worker" in sys.argv:
        try:
            from deepseek_http_worker import tls_context
            connection = http.client.HTTPSConnection("api.deepseek.com", timeout=15,
                                                     context=tls_context())
            try:
                connection.request("GET", "/models", headers={"Accept": "application/json"})
                status = connection.getresponse().status
                print(json.dumps(dict(http_status=status, authenticated=False)))
                return 0 if status == 401 else 1
            finally:
                connection.close()
        except Exception as error:
            # Only a builtin exception type name, never its message or headers.
            print(json.dumps(dict(error_type=type(error).__name__)))
            return 1
    if "--inside" in sys.argv:
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LD_LIBRARY_PATH") if k in os.environ}
        # -I removes the source directory; use the explicit trusted script path.
        code = ("import runpy,sys; sys.path.insert(0,sys.argv[1]); path=sys.argv[2]; "
                "sys.argv=[path,'--worker']; runpy.run_path(path,run_name='__main__')")
        try:
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                     str(Path(__file__).resolve().parent), str(Path(__file__).resolve())],
                                    env=env, capture_output=True, timeout=25)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(error_type="ParentTimeout")))
            return 1
        # This child never handles a credential; still suppress raw stderr.
        print(result.stdout.decode("utf-8", errors="replace").strip())
        print(json.dumps(dict(worker_exit=result.returncode, stderr_present=bool(result.stderr))))
        return result.returncode
    from hecate_python_env import VENV, enter_nix
    command = (f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
               f'{shlex.quote(str(VENV / "bin/python"))} '
               f'{shlex.quote(str(Path(__file__).resolve()))} --inside')
    return enter_nix(command, seconds=60)


if __name__ == "__main__":
    raise SystemExit(main())
