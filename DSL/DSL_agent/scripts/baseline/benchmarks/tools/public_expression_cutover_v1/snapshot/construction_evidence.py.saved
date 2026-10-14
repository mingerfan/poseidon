"""Finite witness failures are not claims of semantic impossibility."""
class ConstructionEvidenceError(ValueError):
    def __init__(self, message, *, feature, reason, observed_positions=0, attempts=0):
        super().__init__(message)
        self.evidence = dict(feature=feature, reason=reason,
                            observed_positions=observed_positions, attempts=attempts)
