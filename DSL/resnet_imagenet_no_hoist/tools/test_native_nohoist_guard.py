"""CPU-only guard checks; does not start a GPU process."""
from unittest.mock import patch
from run_native_nohoist import idle_gpu


for free, tasks, allowed in ((32498, '', True), (27000, '', False),
                              (32498, 'GPU-example\n', False)):
    with patch('subprocess.check_output', side_effect=[f'GPU-example, {free}\n', tasks]):
        try:
            assert idle_gpu(2) == 'GPU-example'
            assert allowed
        except RuntimeError:
            assert not allowed
print('PASS: idle GPU accepted; occupied GPU and insufficient headroom rejected')
