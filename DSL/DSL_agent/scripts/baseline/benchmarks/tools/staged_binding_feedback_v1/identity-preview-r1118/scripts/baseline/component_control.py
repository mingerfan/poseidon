"""Cancellation and callback adapters over the shared bounded feedback policy."""
class QualificationCancelled(BaseException):
    """Crosses subprocess.run, which kills its child before unwinding."""
    pass


class PreciseFeedbackControl:
    """CLI adapter; generation and HTTP budgets remain with existing owners.

    Does not run a provider, compiler or backend. The trusted evaluator retains
    numerical acceptance and persists all original attempt evidence.
    """
    def __init__(self, report, *, stopped_response=None):
        from application_diagnostics import RepairDecision, PRECISE
        self.report = report
        self.policy = PRECISE
        self.decision = RepairDecision()
        self.stopped_response = stopped_response
        self.last_diagnostic = None

    def before_generation(self, index):
        return None  # run_feedback_loop and Provider own existing finite budgets.

    def before_evaluation(self, raw, index):
        stop = self.decision.before_candidate(raw)
        if stop:
            self.report["feedback_stop"] = dict(index=index, failure=self.decision.last_failure)
            if self.stopped_response is not None:
                self.stopped_response(index, raw)
        return stop

    def evaluate(self, evaluator, raw, index):
        from application_diagnostics import attach
        from component_contract import failure
        from candidate_contract import strict_json
        feedback = evaluator(raw, index)
        if feedback.get("status") == "passed":
            return feedback
        attempts = self.report.get("attempts", [])
        attempt = next((a for a in reversed(attempts) if a.get("index") == index), {})
        result = dict(status="failed", public_feedback=feedback,
            failure=failure(attempt.get("failure_layer", feedback.get("layer", "pipeline")),
                            attempt.get("diagnostic", "")))
        if "compiler_diagnosis" in attempt:
            result["compiler_diagnosis"] = attempt["compiler_diagnosis"]
        try:
            candidate = strict_json(raw)
        except (ValueError, TypeError):
            candidate = None
        result = attach(result, candidate=candidate)
        self.last_diagnostic = result["structured_diagnostic"]
        if attempt:
            attempt["structured_diagnostic"] = self.last_diagnostic
        return result["public_feedback"]

    def after_evaluation(self, feedback, index):
        from application_diagnostics import from_result
        diagnostic = self.last_diagnostic or from_result(dict(public_feedback=feedback))
        stop = self.decision.after_failure(diagnostic)
        if stop:
            self.report["feedback_stop"] = dict(index=index, failure=self.decision.last_failure)
        self.last_diagnostic = None
        return stop
