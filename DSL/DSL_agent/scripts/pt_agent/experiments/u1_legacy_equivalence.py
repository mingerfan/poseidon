"""U1 §7.1: compare current intake against pre-U1 code and frozen programs."""

from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch


def historical_module(filename, name, package):
    from ..paths import REPO
    source = subprocess.check_output(['git', 'show', '0189ce11:DSL/DSL_agent/scripts/pt_agent/'+filename],
                                     cwd=REPO, text=True)
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__).resolve().parents[1]/filename)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def compare_model(name, models_dir, frozen_root, output=None):
    import numpy as np
    from ..analyzer import calibrate
    from ..cost import estimate
    from ..data.samples import Samples, legacy_cifar
    from ..dsl.parser import parse
    from ..intake.probe import import_torch
    from ..models.builders import load
    from ..planners import agent_tools, confirm
    old_zoo = historical_module('models/zoo.py', 'pt_agent.models._u1_old_zoo', 'pt_agent.models')
    old_aten = historical_module('intake/aten_frontend.py', 'pt_agent.intake._u1_old_aten', 'pt_agent.intake')
    old_tools = historical_module('planners/agent_tools.py', 'pt_agent.planners._u1_old_tools', 'pt_agent.planners')
    old_confirm = historical_module('planners/confirm.py', 'pt_agent.planners._u1_old_confirm', 'pt_agent.planners')
    old_confirm.validate = old_tools.validate
    old_model = old_zoo.get(name, models_dir)
    before = old_aten.import_module_aten(old_model.module, (old_model.example_input,), name)
    _, legacy_before = legacy_cifar(before, normalization=old_model.normalization)
    old_arrays, labels = legacy_before.get('calibration')
    original_images = old_arrays[before.inputs[0]]
    # These are the exact tensors the original module ordinarily receives. The
    # importer receives no flatten/normalization/task metadata from the user.
    raw = Samples('tensor', [{'input': original_images[i:i+1]} for i in range(len(original_images))],
                  [str(i) for i in range(len(original_images))], labels)
    imported = import_torch(load(name, models_dir), raw, name=name, source='pt_agent.models.builders:'+name)
    after = imported.info.graph
    info, source = legacy_cifar(after, normalization=old_model.normalization)
    with tempfile.TemporaryDirectory() as temporary:
        a, b = Path(temporary)/'old', Path(temporary)/'new'
        before.save(a); after.save(b)
        if (a/'graph.json').read_text() != (b/'graph.json').read_text():
            raise AssertionError(name+': graph.json text changed')
    if before.weights.keys() != after.weights.keys() or any(
            not np.array_equal(before.weights[key], after.weights[key]) for key in before.weights):
        raise AssertionError(name+': graph weights changed')
    arrays, new_labels = source.get('calibration')
    if not np.array_equal(arrays[after.inputs[0]], original_images) or not np.array_equal(labels, new_labels):
        raise AssertionError(name+': calibration tensors changed')
    old_calibration = calibrate(before, old_arrays)
    new_calibration = calibrate(after, arrays)
    if old_calibration != new_calibration:
        raise AssertionError(name+': calibration statistics changed')
    if output is not None:
        info.save(output)
        (output/'calibration.json').write_text(json.dumps(new_calibration,indent=2,allow_nan=False)+'\n')
    old_ctx = old_tools.build_context(before, old_calibration, normalization=old_model.normalization)
    ctx = agent_tools.build_context(info, new_calibration, samples=source, normalization=old_model.normalization)
    for field in ('val_images','val_exact_logits','val_labels','probe_images'):
        if not np.array_equal(getattr(old_ctx, field), getattr(ctx, field)):
            raise AssertionError(name+': context '+field+' changed')
    for field in ('val_exact_top1','needed_margin','act_tensors','model_name','profile','val_split','normalization'):
        if getattr(old_ctx, field) != getattr(ctx, field):
            raise AssertionError(name+': context '+field+' changed')
    if any(not np.array_equal(a,b) for a,b in zip(old_ctx.probe_act_inputs,ctx.probe_act_inputs)):
        raise AssertionError(name+': activation probes changed')
    text = (Path(frozen_root)/name/'program.ptdsl').read_text()
    program = parse(text)
    validations = {}
    for engine in ('mock','tensor'):
        old = old_tools.validate(program, old_ctx, engine=engine, cache=False)
        new = agent_tools.validate(program, ctx, engine=engine, cache=False)
        if old != new:
            differences = [key for key in old.keys() | new.keys() if old.get(key) != new.get(key)]
            raise AssertionError(name+': '+engine+' validation changed: '+str(differences))
        validations[engine] = new
        print('LEGACY '+name+' '+engine+' equal',flush=True)
    old_cost = estimate(program, before.weights)
    new_cost = estimate(program, after.weights)
    if old_cost != new_cost:
        raise AssertionError(name+': estimate changed')
    # Reuse the full real Mock results above while checking the complete outward
    # feedback. No additional full validation is needed for formatting proof.
    with patch.object(old_tools, 'validate', return_value=validations['mock']), \
            patch.object(agent_tools, 'validate', return_value=validations['mock']):
        old_fb = old_tools.evaluate(text, old_ctx)
        new_fb = agent_tools.evaluate(text, ctx)
    if json.dumps(old_fb,sort_keys=True,allow_nan=False) != json.dumps(new_fb,sort_keys=True,allow_nan=False):
        raise AssertionError(name+': labelled feedback text changed')
    candidate = confirm.Candidate(text,'frozen',0,new_cost['latency_pred_us'],new_cost['boots_pred'])
    previous = old_confirm.confirm_select([candidate],old_ctx)
    current = confirm.confirm_select([candidate],ctx)
    for result in (previous,current):
        result['selected'] = asdict(result['selected']) if result['selected'] is not None else None
    if previous != current:
        raise AssertionError(name+': confirm_select changed')
    result = dict(model=name,graph_text_equal=True,weights_equal=True,calibration_equal=True,
                  context_equal=True,validate_equal=dict(mock=True,tensor=True),estimate_equal=True,
                  confirm_equal=True,feedback_text_equal=True,validation_images=len(ctx.val_images),
                  confirmation_candidates=1,report_images_read=0,
                  metrics={key:validations['mock'].get(key) for key in ('top1','exact_top1','acc_drop_pp','agree','n_diverged')})
    del old_ctx,ctx,before,after,original_images,old_arrays,arrays
    gc.collect()
    return result


def run(args):
    from ..paths import RESULTS_DIR
    rows = []
    for name in ('resnet20','resnet20_silu','cnn_plain','mlp'):
        if args.only and name not in args.only.split(','):
            continue
        rows.append(compare_model(name, args.models_dir, args.reference/'models', args.out/name))
        (args.out/'LEGACY.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    return rows
