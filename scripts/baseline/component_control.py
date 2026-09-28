"""Cancellation crosses Python subprocess.run, which kills its child before unwinding."""
class QualificationCancelled(BaseException):
    pass
