"""Present workflow names without rewriting stored source/version identifiers."""

import re


WORKFLOW_NAMES = {
    "1": "Overview", "2": "Data_Integration", "3": "Spectral_Analytics",
    "4": "Predictive_Modeling", "5": "Process_Monitoring",
    "6": "Multivariate_Monitoring", "7": "Model_Reliability", "8": "Model_Deployment",
}
_PHASE = re.compile(r"\bphases?\s+([1-8])(?:/([1-8]))?\b", re.IGNORECASE)


def workflow_label(value):
    """For widget format_func: the selected value remains its original value.

    Spaces after 'Phase' are intentional: artifact versions such as
    phase5-demo-* and existing phase_2_learning_guide.md paths stay unchanged.
    Scientific uses such as Fourier phase and process phase are also preserved.
    """
    def replace(match):
        first = WORKFLOW_NAMES[match.group(1)]
        return first if match.group(2) is None else f"{first} / {WORKFLOW_NAMES[match.group(2)]}"
    return _PHASE.sub(replace, str(value))
