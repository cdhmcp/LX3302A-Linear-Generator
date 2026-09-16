"""Portable core for Microchip inductive-sensor footprint generation."""

from .diagnostics import Diagnostic, DiagnosticSeverity, ValidationReport
from .engine import GENERATORS, GenerationBlocked, LinearSensorDefinition
from .models import GenerationRequest, LinearSensorConfig, OutputConfig, default_request

__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "GENERATORS",
    "GenerationBlocked",
    "GenerationRequest",
    "LinearSensorConfig",
    "LinearSensorDefinition",
    "OutputConfig",
    "ValidationReport",
    "default_request",
]
