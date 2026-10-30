"""Turn finished runs into a provider comparison that says what it can prove.

    searchforge-report outputs/<run> [outputs/<run2> ...]
"""

from searchforge.analysis.report import build_report, render_markdown
from searchforge.analysis.runs import load_run

__all__ = ["build_report", "load_run", "render_markdown"]
