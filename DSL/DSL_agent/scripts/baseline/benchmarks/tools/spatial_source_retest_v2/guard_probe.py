"""Developer guard probe, never constructs a Provider or backend."""
import argparse,json,time
import run
parser=argparse.ArgumentParser()
parser.add_argument("--probe",action="store_true");parser.add_argument("--execute",action="store_true")
args=parser.parse_args()
if args.execute:
 print(json.dumps(dict(mode="owned_nonancestor_fixture")),flush=True);time.sleep(5)
else:
 run.no_active();print(json.dumps(dict(active_guard="passed",api_calls=0)),flush=True)
