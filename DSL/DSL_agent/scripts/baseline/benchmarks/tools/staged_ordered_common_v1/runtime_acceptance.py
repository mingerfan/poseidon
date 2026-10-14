"""Legacy common-call wrapper using the shared explicit-scope driver."""
import argparse,fcntl,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
import runtime_scope as scope
from runtime_driver import Driver
HERE=scope.HERE;ROOT=scope.ROOT;BASE=scope.BASE;WORK=scope.WORK;RESULTS=scope.RESULTS
_driver=Driver(scope,Path(__file__).resolve())
allowed_env=_driver.allowed_env
unstarted_attempt=_driver.unstarted_attempt
spawn_namespace=_driver.spawn_namespace
outer=_driver.outer
namespace=_driver.namespace
mount_page=_driver.mount_page
wait_namespace=_driver.wait_namespace
mount_layer=_driver.mount_layer
boundary=_driver.boundary
inside=_driver.inside
one=_driver.one
if __name__=="__main__":raise SystemExit(_driver.main())
