"""Verify historical analyzer provenance without freezing the live operator set."""
import hashlib
from pathlib import Path
import re
from audit_agent_lineage import read,require

def verify_sources(report,root):
    root=Path(root)
    for name,digest in report['analysis_source_hashes'].items():
        current,current_hash=read(root/name,root)
        if current_hash==digest:continue
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),'Invalid archived source digest')
        _,archived_hash=read(root/'history'/(digest+'.py'),root)
        require(archived_hash==digest,'Historical analyzer source hash mismatch')

def compare_historical(report,current):
    # Historical observations must agree verbatim on the old domain. New
    # unsupported/unobserved capabilities must not erase old missing entries.
    old_ops={r['operator'] for r in report['operator_matrix']}
    for key in ('graph_case_details','legacy_cases','feature_matrix'):
        require(report[key]==current[key],'Historical observations changed: '+key)
    rows=[r for r in current['operator_matrix'] if r['operator'] in old_ops]
    require(rows==report['operator_matrix'],'Historical operator observations changed')
    require([op for op in current['missing_graph_operators'] if op in old_ops]==report['missing_graph_operators'],
            'Historical missing coverage changed')
