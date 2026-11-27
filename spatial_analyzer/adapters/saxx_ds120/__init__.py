"""Schlanker Saxx-DS120-Adapter: filter_export.json -> Edit-JSON fuer den externen
Candidate Builder 0.15.0-qa. Keine .datdsp-Erzeugung, keine Hardware-I/O."""

from spatial_analyzer.adapters.saxx_ds120.mapper import (
    MAPPER_VERSION,
    AdapterBlockedError,
    MappingError,
    MappingResult,
    compute_mapping_input_sha256,
    compute_mapping_output_sha256,
    map_export_to_edit,
    verify_mapping,
)
from spatial_analyzer.adapters.saxx_ds120.preview import build_preview, render_preview
from spatial_analyzer.adapters.saxx_ds120.quantization import (
    QuantizationError,
    SaxxAdapterError,
    load_lookup_tables,
    quantize,
)
from spatial_analyzer.adapters.saxx_ds120.validation import (
    ValidationError,
    assert_opaque_preserved,
    validate_edit_json,
)

__all__ = [
    "MAPPER_VERSION", "AdapterBlockedError", "MappingError", "MappingResult", "QuantizationError",
    "SaxxAdapterError", "ValidationError", "assert_opaque_preserved", "build_preview",
    "compute_mapping_input_sha256", "compute_mapping_output_sha256", "load_lookup_tables",
    "map_export_to_edit", "quantize", "render_preview", "validate_edit_json", "verify_mapping",
]
