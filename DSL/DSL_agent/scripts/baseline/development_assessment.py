"""Explicit name for the legacy generate/qualify/feedback development workflow."""


def run_development(application, task, provider):
    """Preserve the legacy lifecycle; never count its repaired result as delivery."""
    result = application.run(task, provider)
    return dict(mode='development_assessment', first_output_acceptance_eligible=False,
                legacy_application_result=result)
