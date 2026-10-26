"""Build a relocatable package from parsed PT-DSL and its public weights."""

import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from .codegen import generate
from .. import toolchain
from ..backends import CountBackend
from ..cost import PROFILES
from ..dsl.ir import WeightRef
from ..dsl.typecheck import check
from ..kernels import FlatLayout, MuxLayout
from ..kernels.layout import layout_blocks, layout_record
from ..kernels.poly import cheb_eval, cheb_fit, minimax_coeffs
from ..kernels import approx
from ..lower import _pack_layout, lower


SOURCE = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent/"templates"
CORE_SOURCES = {"base.py": "backends/base.py", "refresh.py": "backends/refresh.py",
                **{name+".py": "kernels/"+name+".py" for name in ("layout", "slots", "conv", "pool", "linear", "multi", "cnn")}}


def display_name(name):
    """Capitalize each generic letter/number token: resnet20 -> ResNet20."""
    return "".join(token.capitalize() if token.isalpha() else token
                   for token in re.findall(r"[A-Za-z]+|[0-9]+", re.sub(r"(?i)net", " net ", name)))


def compile_template(mode):
    substitutions = {"/hecate-opt": "{hecate_opt}", "/out/candidate_trace.mlir": "{input}",
                     "/out/boot-compiler-config.json": "{package}/config/compiler-config.json",
                     "/out/lowered.mlir": "{output_dir}/lowered.mlir"}
    result = []
    for arg in toolchain.compile_argv(mode, "qwen48", Path(".")):
        for old, new in substitutions.items():
            arg = arg.replace(old, new)
        if arg.startswith("/") or re.search(r"=/", arg):
            raise ValueError(f"Unexpected absolute compiler path: {arg}")
        result.append(arg)
    return result


def managed_compile_template():
    substitutions = {'/hecate-opt': '{hecate_opt}', '/out/managed.mlir': '{input}',
                     '/out/boot-compiler-config.json': '{package}/config/compiler-config.json',
                     '/out/': '{output_dir}/'}
    result = []
    for command in toolchain.managed_compile_argv('dacapo-depth-dp', 'qwen48', Path('.')):
        argv = []
        for arg in command:
            arg = arg.replace('{', '{{').replace('}', '}}')
            for old, new in substitutions.items():
                arg = arg.replace(old, new)
            argv.append(arg)
        result.append(argv)
    return result


def frontend_preprocess_template():
    result = []
    substitutions = {'/hecate-opt':'{hecate_opt}', '/out/raw.earth.mlir':'{input}',
                     '/out/boot-compiler-config.json':'{package}/config/compiler-config.json',
                     '/out/':'{output_dir}/'}
    for arg in toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',Path('.')):
        arg = arg.replace('{','{{').replace('}','}}')
        for old,new in substitutions.items():
            arg = arg.replace(old,new)
        result.append(arg)
    return result


def _management_files(program, weights, code, module, package):
    """Ship the same manager and the sole placement implementation, without SDK imports."""
    profile = toolchain.management_profile()
    record = dict(mode=program.header.manage, profile=profile, cuts=None, boots=None, values=[],
                  source_lines={}, input_dag='pending-frontend-preprocess')
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            op = program.producer(node.targets[0].id)
            if op is not None:
                record['source_lines'][str(node.lineno)] = dict(id=op.id, lineno=op.lineno)
    _dump(package/'config/management.json', record)
    _dump(package/'config/managed_compile_argv.json', managed_compile_template())
    _dump(package/'config/frontend_preprocess_argv.json', frontend_preprocess_template())
    target = module/'core/manage'
    target.mkdir()
    for source in sorted((SOURCE/'manage').glob('*.py')):
        text = source.read_text().replace('from ..backends.base import', 'from ..base import')
        (target/source.name).write_text(text)
    # placement imports DagBackend solely for its annotation. The standalone
    # package uses the recording backend's duck-typed view of the identical DAG.
    (module/'core/placement.py').write_text((SOURCE/'placement.py').read_text().replace(
        'from .backends.dag import DagBackend', 'from typing import Any as DagBackend'))


def _check_imports(text, label):
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.level:
            target = node.module or ""
            if node.level != 1 or target.split(".")[0]+".py" not in CORE_SOURCES:
                raise ValueError(f"Non-core import in {label}: {target}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            for module in modules:
                root = module.split(".")[0]
                if root != "numpy" and root not in sys.stdlib_module_names:
                    raise ValueError(f"Non-core import in {label}: {module}")


def _dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_manifest(package):
    """Write the distribution manifest last, including generated validation descriptions."""
    config = json.loads((package/"config/model.json").read_text())
    validation = config.get("validation", {})
    native = bool(validation.get("native_compiled", False))
    state = (f"已原生编译，自举 {validation['boots']} 次；CKKS 和 GPU 尚未验证"
             if native else "原生编译、CKKS 和 GPU 待验证")
    manifest = dict(package=package.name, date=datetime.now(timezone.utc).date().isoformat(), config=config,
                    scope="Complete PT-DSL program; structured kernels; external public NPZ weights",
                    validation=state, pretrained_weights_included=False, native_compiled=native,
                    ckks_validated=False, gpu_validated=False, files={})
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            content = path.read_bytes()
            manifest["files"][path.relative_to(package).as_posix()] = dict(
                bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    _dump(package/"manifest.json", manifest)


def _readme(program, interface, approximations, validation):
    title = display_name(program.name)+"_PTAgent_DSL"
    rows = ["| 值名 | 函数 | 方法 | 阶数 | 定义域 |", "|---|---|---|---|---|"]
    for name, item in approximations.items():
        degree = (f"deg0={item['deg0']}, iters={item['iters']}" if item['method']=='newton'
                  else item.get('degrees',item.get('deg','exact')))
        rows.append(f"| {name} | {item['fn']} | {item['method']} | "
                    f"{degree} | {item.get('domain', '')} |")
    inputs = "; ".join(f"{item['name']} {item['shape']} {item['layout']}" for item in interface['inputs'])
    outputs = "; ".join(f"{item['name']} {item['shape']} {item['layout']}" for item in interface['outputs'])
    state = (f"已原生编译，自举 {validation['boots']} 次；CKKS 和 GPU 尚未验证。" if
             validation.get("native_compiled") else "原生编译、CKKS 和 GPU 待验证。")
    return f'''# {title}

## 1. 模型内容和接口

程序 `{program.name}`，slots={program.header.slots}，刷新方式 `{program.header.refresh}`。
输入：{inputs}。
输出：{outputs}。
接口快照见 `interfaces/default.json`。每个接口项列出全部密文下标及分块布局。

## 2. 近似参数

{chr(10).join(rows)}

全部 float64 系数已保存在 `config/approximations.json`，运行时直接读取。

## 3. 文件入口

`{program.name}/model.py` 保留逐层调用；`ops.py` 调用 `core/` 中的算术内核。
`trace_model.py` 导出 MLIR，`compile_model.py` 使用固定命令模板编译。
其余入口负责打包输入、NumPy 自检、解包输出、比较输出和验证清单。

## 4. 依赖

Python ≥3.10、NumPy。trace 还需调用方环境中的 hecate 前端及 torch。
调用方设置 `HECATE`、`HECATE_BUILD` 和指向 hecate Python 包父目录的 `PYTHONPATH`。
所有配置及内核路径相对包根目录解析。

## 5. 外部权重 NPZ

权重保存在包外。键名为 PyTorch state_dict 名称，只包含 `config/weight_schema.json` 中列出的张量。
BN 使用 weight、bias、running_mean、running_var 四个键。
可在模型已有的环境中导出：

```python
import json
import numpy as np
from pathlib import Path
schema = json.loads(Path("config/weight_schema.json").read_text())
state = model.state_dict()
np.savez("weights.npz", **{{name: state[name].detach().cpu().numpy().astype(np.float64) for name in schema}})
```

包只读取数字 NPZ。输入是模型接收的张量，归一化由调用方在包外完成。

## 6. 命令

在包根目录执行，权重和输入路径由调用方提供：

```bash
python verify_manifest.py
python trace_model.py --plan
python prepare_inputs.py --inputs inputs.npy --output input_slots.npy
python run_numpy.py --weights ../weights/{program.name}.npz --inputs inputs.npy --output reference.npy
python trace_model.py --weights ../weights/{program.name}.npz --output-dir trace
python compile_model.py --hecate-opt "$HECATE_BUILD/bin/hecate-opt" --input trace/trace_model.mlir --output-dir compiled
python unpack_outputs.py --signature trace/signature.json --slots decrypted_slots.npy --output actual.npy
python check_outputs.py --actual actual.npy --reference reference.npy
```

trace 和 compile 均要求新的输出目录。多个输入或输出使用按接口名称索引的 NPZ。
比较条件为 `1e-5 + 1e-4*abs(reference)`，同时报告 argmax 一致率。

## 7. 验证状态

{state} 详见 manifest.json 的 validation 标志。

## 8. 生成说明

本包由 pt_agent 从 `program.ptdsl` 自动生成，请勿手改；修改 PT-DSL 后重新导出。
'''


def export_package(program_text, program, weights, out_dir, *, run_info=None) -> Path:
    # Run the frozen lowering checks before writing any package file.
    _, output_layouts = lower(program, weights, CountBackend(program.header.slots, boot_out_level=5))
    code = generate(program)
    checked = check(program, {name: value.shape for name, value in weights.items()})
    if not checked.ok:
        raise ValueError("PT-DSL typecheck failed: "+str(checked.diags))
    core_texts = {name: (SOURCE/source).read_text() for name, source in CORE_SOURCES.items()}
    for name, text in core_texts.items():
        _check_imports(text, name)
    info = dict(run_info or {})
    directory = Path(out_dir)
    package = directory/(display_name(program.name)+"_PTAgent_DSL")
    weights_path = directory/"weights"/(program.name+".npz")
    if package.exists() or weights_path.exists():
        raise FileExistsError("Use a new package and weights destination")
    mode = "pars" if program.header.refresh == "agent" else "dacapo-depth-dp"
    config = dict(name=program.name, slots=program.header.slots, refresh=program.header.refresh,
                  placement=program.header.placement, compile_mode=mode, boot_profile="qwen48",
                  input_shapes=[list(param.shape) for param in program.params],
                  output_shapes=[list(checked.shapes[name]) for name in program.outputs],
                  validation=info.get("validation", {}))
    if program.header.refresh == "agent":
        config.update({key: PROFILES["qwen48"][key] for key in ("input_level", "boot_in_level", "boot_out_level")})
    approximations = {}
    names = set()
    for op in program.ops:
        for value in op.attrs.values():
            if isinstance(value, WeightRef):
                names.add(value.name)
        if op.kind in ("conv2d", "batchnorm") and op.attrs["bn"] is not None:
            names.update(op.attrs["bn"]+"."+suffix for suffix in ("weight", "bias", "running_mean", "running_var"))
        if op.kind in ("act", "maxpool"):
            a = op.attrs
            fn = a.get("fn", "relu")
            entry = dict(op_id=op.id, lineno=op.lineno, fn=fn, method=a["method"])
            if a["method"] == "cheb":
                entry.update(deg=a["deg"], domain=a["domain"],
                             coefficients=cheb_fit(fn, a["deg"], *a["domain"]).tolist())
            elif a["method"] == "minimax_comp":
                entry.update(degrees=a["degrees"], domain=a["domain"],
                             coefficients=[value.tolist() for value in minimax_coeffs(tuple(a["degrees"]))])
            elif a['method'] in approx.NEW_METHODS:
                coefficients=approx.fitted_coefficients(a)
                entry.update(domain=a['domain'],coefficients=(
                    [value.tolist() for value in coefficients] if a['method']=='composite' else coefficients.tolist()))
                if a['method']=='remez':
                    entry['deg']=a['deg']
                elif a['method']=='composite':
                    entry.update(degrees=a['degrees'],eps=a['eps'])
                else:
                    entry.update(deg0=a['deg0'],iters=a['iters'],
                                 iteration='inv_y_2_minus_xy' if fn=='inv' else 'rsqrt_x_half')
            approximations[op.outputs[0]] = entry
    schema = {name: list(weights[name].shape) for name in sorted(names)}

    def interface_entry(name, shape, lay, index):
        blocks = layout_blocks(lay)
        return dict(name=name, shape=list(shape), ciphertext=index,
                    ciphertexts=list(range(index, index+len(blocks))),
                    blocks=[dict(start=start, end=end, ciphertext=index+i, layout=layout_record(block))
                            for i, (start, end, block) in enumerate(blocks)], layout=layout_record(lay))

    inputs = []
    input_index = 0
    for param in program.params:
        ops = [op for op in program.ops if op.kind == "pack" and op.inputs == (param.name,)]
        if len(ops) != 1:
            raise ValueError("each parameter requires one pack operation")
        lay = _pack_layout(ops[0], param.shape, program.header.slots)
        inputs.append(interface_entry(param.name, param.shape, lay, input_index))
        input_index += len(layout_blocks(lay))
    outputs, output_index = [], 0
    for name, lay in zip(program.outputs, output_layouts):
        outputs.append(interface_entry(name, checked.shapes[name], lay, output_index))
        output_index += len(layout_blocks(lay))
    interface = dict(format="pt-agent-dsl-v1", function="_hecate_"+program.name, slots=program.header.slots,
                     inputs=inputs, outputs=outputs)
    template = compile_template(mode)
    package.mkdir(parents=True)
    for path in sorted(TEMPLATES.rglob("*.py")):
        relative = path.relative_to(TEMPLATES)
        parts = (program.name, *relative.parts[1:]) if relative.parts[0] == "_package" else relative.parts
        target = package.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    module = package/program.name
    (module/"model.py").write_text(code)
    core = module/"core"
    core.mkdir()
    for name, text in core_texts.items():
        (core/name).write_text(text)
    (core/"cheb.py").write_text('"""Copied Chebyshev evaluator."""\n\nimport numpy as np\n'
                                'from .layout import KernelError\nfrom .slots import accumulate\n\n\n'+inspect.getsource(cheb_eval))
    new_approximations=any(a['method'] in approx.NEW_METHODS for a in approximations.values())
    if new_approximations:
        (core/'approx.py').write_text('"""Coefficient-only U5 evaluator; fitting is external."""\n'
            'from .cheb import cheb_eval\n\n'+inspect.getsource(approx.evaluate))
    (core/"__init__.py").write_text('"""Arithmetic used by this package."""\n'
        'from .layout import FlatLayout, MuxLayout, KernelError, pack_flat, pack_mux, unpack_flat, unpack_mux\n'
        'from .refresh import RefreshBackend\nfrom .cheb import cheb_eval\n')
    (package/"program.ptdsl").write_text(program_text)
    _dump(package/"config/model.json", config)
    _dump(package/"config/approximations.json", approximations)
    _dump(package/"config/weight_schema.json", schema)
    _dump(package/"config/compile_argv.json", template)
    _dump(package/"interfaces/default.json", interface)
    _management_files(program, weights, code, module, package)
    for name in ("compiler-config.json", "operator-spec.json"):
        shutil.copyfile(toolchain.BOOT_CONFIG_DIR/name, package/"config"/name)
    git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip())
    tag = subprocess.run(["git", "tag", "--list", "pt-agent-freeze-v1"], cwd=SOURCE,
                         capture_output=True, text=True, check=True).stdout.strip()

    def source_path(value):
        return None if value is None else os.path.relpath(value, directory) if Path(value).is_absolute() else str(value)

    copied = {str(Path(program.name)/"core"/name): "DSL/DSL_agent/scripts/pt_agent/"+source
              for name, source in CORE_SOURCES.items()}
    copied[str(Path(program.name)/"core/cheb.py")] = "DSL/DSL_agent/scripts/pt_agent/kernels/poly.py:cheb_eval"
    if new_approximations:
        copied[str(Path(program.name)/'core/approx.py')]='DSL/DSL_agent/scripts/pt_agent/kernels/approx.py:evaluate'
    _dump(package/"provenance.json", dict(ptdsl_source=source_path(info.get("program_path", "program.ptdsl")),
          pt_agent_git_rev=git_rev, dirty=dirty, freeze_tag=tag or None,
          agent_run_dir=source_path(info.get("run_dir")), copied_sources=copied))
    (package/"README.md").write_text(_readme(program, interface, approximations, config["validation"]))
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, **{name: np.asarray(weights[name], dtype=np.float64) for name in schema})
    write_manifest(package)
    return package
