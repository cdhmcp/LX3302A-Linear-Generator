"""Structured validation messages shared by the core, CLI, and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class Diagnostic:
    """One actionable result from configuration or geometry validation."""

    code: str
    severity: DiagnosticSeverity
    message: str
    field: str | None = None
    suggestion: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Validation results with convenience properties for generation policy."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(
            item for item in self.diagnostics if item.severity is DiagnosticSeverity.WARNING
        )

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(
            item for item in self.diagnostics if item.severity is DiagnosticSeverity.ERROR
        )

    @property
    def blocking(self) -> tuple[Diagnostic, ...]:
        return tuple(
            item for item in self.diagnostics if item.severity is DiagnosticSeverity.BLOCKING
        )

    @property
    def can_generate_normally(self) -> bool:
        return not self.errors and not self.blocking

    @property
    def can_force_generate(self) -> bool:
        return not self.blocking

    def with_diagnostic(self, diagnostic: Diagnostic) -> "ValidationReport":
        return ValidationReport(self.diagnostics + (diagnostic,))
