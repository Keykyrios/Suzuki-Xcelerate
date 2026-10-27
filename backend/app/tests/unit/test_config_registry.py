"""Configuration registry: provenance metadata, ranges, required parameters."""

from __future__ import annotations

import pytest

from app.config.evidence import EvidenceStatus
from app.config.loader import CONFIG_FILES, ConfigBundle, get_config
from app.config.registry import ConfigError, Parameter, ParameterRegistry, load_registry


@pytest.fixture(scope="module")
def cfg() -> ConfigBundle:
    return get_config()


def test_all_config_files_loaded(cfg):
    assert set(cfg.registries) == set(CONFIG_FILES)
    for name, registry in cfg.registries.items():
        assert registry.version, f"{name} has no version"
        assert registry.all_parameters(), f"{name} has no parameters"


def test_every_parameter_carries_full_provenance(cfg):
    for registry in cfg.registries.values():
        for param in registry.all_parameters():
            assert param.name.strip(), f"{registry.source_name}:{param.path} missing name"
            assert param.unit.strip(), f"{registry.source_name}:{param.path} missing unit"
            assert param.source_or_rationale.strip(), (
                f"{registry.source_name}:{param.path} missing source_or_rationale"
            )
            assert isinstance(param.status, EvidenceStatus), (
                f"{registry.source_name}:{param.path} status {param.status} is not a taxonomy label"
            )


def test_every_numeric_parameter_declares_a_range_or_is_unbounded_by_intent(cfg):
    for registry in cfg.registries.values():
        for param in registry.all_parameters():
            if isinstance(param.value, (int, float)) and not isinstance(param.value, bool):
                if param.min is not None and param.max is not None:
                    assert param.min <= param.max
                    assert param.min <= float(param.value) <= param.max


def test_reference_vehicle_parameters_present(cfg):
    required_vehicle = [
        "package.curb_mass_kg",
        "package.passenger_mass_kg",
        "package.max_payload_kg",
        "package.passenger_capacity",
        "package.luggage_mass_per_unit_kg",
        "aerodynamics.drag_coefficient",
        "aerodynamics.frontal_area_m2",
        "resistance.rolling_resistance_coefficient",
        "resistance.air_density_kg_m3",
        "resistance.gravity_m_s2",
        "drivetrain.traction_efficiency",
        "drivetrain.peak_traction_power_kw",
        "regeneration.recovery_fraction",
        "regeneration.max_regen_power_kw",
        "regeneration.path_efficiency",
        "regeneration.min_speed_m_s",
        "auxiliary.auxiliary_load_kw",
        "hvac.base_load_kw",
        "hvac.per_occupied_zone_kw",
        "battery.bev_usable_energy_kwh",
        "battery.rex_usable_energy_kwh",
        "battery.max_charge_power_dc_kw",
        "battery.max_charge_power_ac_kw",
        "battery.charge_efficiency",
        "battery.taper_start_soc",
        "numerics.integration_dt_s",
    ]
    for path in required_vehicle:
        assert path in cfg.vehicle, f"missing vehicle parameter {path}"

    for path in (
        "soc_policy.soc_reserve_min",
        "soc_policy.soc_operating_min",
        "soc_policy.soc_operating_max",
        "state_integrity.stale_after_s",
        "evidence.default_model_status",
        "traffic.min_speed_factor",
    ):
        assert path in cfg.policy, f"missing operating policy parameter {path}"


def test_soc_bands_are_ordered(cfg):
    reserve = float(cfg.op("soc_policy.soc_reserve_min"))
    operating_min = float(cfg.op("soc_policy.soc_operating_min"))
    operating_max = float(cfg.op("soc_policy.soc_operating_max"))
    assert 0.0 < reserve < 1.0
    assert 0.0 < operating_min < operating_max <= 1.0
    assert operating_min <= reserve <= operating_max


def test_hash_is_deterministic_and_content_addressed(cfg, tmp_path):
    other = ConfigBundle(root=cfg.root)
    assert cfg.hash == other.hash
    assert len(cfg.hash) == 64
    # The hash must change when any parameter value changes.
    registry = load_registry(cfg.root / "config" / "vehicle.yaml")
    original = registry.content_hash()
    registry._parameters["numerics.integration_dt_s"].value = 0.5
    assert registry.content_hash() != original


def test_snapshot_exposes_every_parameter_with_source_file(cfg):
    snapshot = cfg.snapshot()
    assert len(snapshot) == sum(len(r.all_parameters()) for r in cfg.registries.values())
    assert all("file" in entry and "status" in entry for entry in snapshot)
    assert {entry["file"] for entry in snapshot} == set(CONFIG_FILES)


def test_registry_rejects_incomplete_records(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(
        "version: 1\nparameters:\n  thing:\n    name: Thing\n    value: 1\n    unit: s\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_registry(path)


def test_registry_rejects_value_outside_declared_range(tmp_path):
    path = tmp_path / "range.yaml"
    path.write_text(
        "version: 1\nparameters:\n"
        "  thing:\n    name: Thing\n    value: 10\n    unit: s\n"
        "    status: SCENARIO_ASSUMPTION\n    source_or_rationale: test\n    min: 0\n    max: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_registry(path)


def test_registry_rejects_bad_evidence_label(tmp_path):
    path = tmp_path / "label.yaml"
    path.write_text(
        "version: 1\nparameters:\n"
        "  thing:\n    name: Thing\n    value: 1\n    unit: s\n"
        "    status: VIBES\n    source_or_rationale: test\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_registry(path)


def test_parameter_requires_non_empty_provenance():
    with pytest.raises(Exception):
        Parameter(
            path="x.y", name="X", value=1, unit="s",
            status="SCENARIO_ASSUMPTION", source_or_rationale="   ",
        )


def test_unknown_parameter_raises_config_error(cfg):
    with pytest.raises(ConfigError):
        cfg.vehicle.parameter("no.such.parameter")


def test_registry_walk_is_versioned_per_file(cfg):
    classes = {type(reg) for reg in cfg.registries.values()}
    assert classes == {ParameterRegistry}