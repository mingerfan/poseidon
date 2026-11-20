#!/usr/bin/env python3
"""Inspect frozen delivery or archive a retained response; no live CLI transport.

The host integration calls delivery_component.generate_once with the existing
approved Provider, shared batch and admission guard. This archival CLI cannot
be used to bypass those controls or claim a new first-generation success.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'baseline'))
from delivery_component import capture_response, read_bytes, verify_delivery
from public_weight_context import read_request
from candidate_contract import request_file_bytes
from public_basis_storage import selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--delivery', required=True)
    inspect.add_argument('--binding', required=True)
    archive = sub.add_parser('capture-retained-response')
    for flag in ('request', 'response', 'mechanism', 'output'):
        archive.add_argument('--'+flag, required=True)
    managed = sub.add_parser('prepare-compiler-managed')
    managed.add_argument('--model',required=True)
    managed.add_argument('--options')
    managed.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    if args.action == 'inspect':
        result = verify_delivery(args.delivery, args.binding)[0]
    elif args.action=='capture-retained-response':
        result = capture_response(read_request(read_bytes(args.request).decode('utf-8')),
            read_bytes(args.response).decode('utf-8'), args.output,
            mechanism=json.loads(read_bytes(args.mechanism)))
    else:
        from compiler_managed_delivery import prepare
        from delivery_component import canonical,write_once
        result=prepare(json.loads(read_bytes(args.model)),
            json.loads(read_bytes(args.options))if args.options else None)
        write_once(Path(args.output),request_file_bytes(result) if selected(result) else canonical(result))
        result=dict(request_id=result['request_id'],request=str(Path(args.output).absolute()),
            mode='computational_DSL_delivery',ciphertext_state_owner='compiler',generation_or_evaluation_performed=False)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
