"""Regression coverage for the point-driven linear sensor footprint generator."""

from __future__ import annotations

import hashlib
import json
import re
import unittest

import linear_sensor_generator as generator


FINGERPRINTS = {
    "1:9:left": "225799134cd7e4dabf52843e29fb0e7ff23aec988a83e79b58682efb5cd52296",
    "1:9:right": "0becb46d21f94f5510d74c802ca1b0f0eab7375e6b1939f4787b24cce6cce6b3",
    "2:9:left": "0e963709d79fc5b17ec75e5d550efbd9bbd546c9ea18eefd3f2fa94561a9ef82",
    "2:9:right": "9d5c73f6923e5457544c3cb643eb3ca027bd1e9c4f25259bbb895fa407ef8a1a",
    "3:9:left": "b6fe114aa53c1895865b477aac836e4ed36c9c18864e17da775c02e42d24c173",
    "3:9:right": "ea01ead7e27121bae2fa2282a1a01a82d7f3df242b05acf0816cfe0f3c99c740",
    "4:12:left": "a1b5f7ece3f8813907017cb55eb29ac0b8b1509b17e562d150dbcd1c06dccde6",
    "4:12:right": "b228f7b375ef5500ada91229b663c68efdb261d343319e13fbc26856626f911f",
    "5:13:left": "070fe766b9ab48f4eb26c5df133ba793ce688eee34599cfe81ba08bf9d9f2e64",
    "5:13:right": "50219408a27059d312ab0282007eb9cee02e3c607a99046e7dcbda03c8950ff6",
}


def point(value: generator.Point) -> list[float]:
    return [value[0], value[1]]


def segment(value: generator.Segment) -> list[list[float]]:
    return [point(value[0]), point(value[1])]


def arc(value: generator.Arc) -> list[list[float]]:
    return [point(value[0]), point(value[1]), point(value[2])]


def geometry_fingerprint(
    cl1: generator.CL1Coil,
    cl2: generator.SecondaryCoil,
) -> str:
    """Hash all copper primitives and via coordinates without point identifiers."""
    payload: dict[str, dict[str, object]] = {}
    for name, coil in (("cl2", cl2), ("cl1", cl1)):
        data: dict[str, object] = {
            "target_segments": [segment(item) for item in coil.target_segments],
            "inner_segments": [segment(item) for item in coil.inner_segments],
            "via_points": [point(coil.points[label]) for label in coil.via_labels],
        }
        if isinstance(coil, generator.CL1Coil):
            data.update(
                crossover_segments=[segment(item) for item in coil.crossover_segments],
                target_arcs=[arc(item) for item in coil.target_arcs],
                inner_arcs=[arc(item) for item in coil.inner_arcs],
            )
        payload[name] = data
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class NamingAndGeometryTests(unittest.TestCase):
    def geometry(
        self,
        turns: int,
        target_y: float,
        fanout_side: str = "left",
        *,
        validate: bool = False,
    ):
        cfg = generator.build_config(
            {
                "number_of_secondary_turns": turns,
                "target_y_mm": target_y,
                "fanout_side": fanout_side,
                "allow_invalid_geometry": not validate,
            }
        )
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        self.assertIsNotNone(cl1)
        self.assertIsNotNone(cl2)
        return cfg, primary, cl1, cl2

    def test_all_turn_counts_and_fanout_sides_preserve_copper_geometry(self):
        for turns, target_y in ((1, 9.0), (2, 9.0), (3, 9.0), (4, 12.0), (5, 13.0)):
            for fanout_side in ("left", "right"):
                with self.subTest(turns=turns, fanout_side=fanout_side):
                    _, _, cl1, cl2 = self.geometry(turns, target_y, fanout_side)
                    key = f"{turns}:{target_y:g}:{fanout_side}"
                    self.assertEqual(geometry_fingerprint(cl1, cl2), FINGERPRINTS[key])

    def test_live_point_keys_and_via_labels_are_descriptive_for_all_turn_counts(self):
        legacy_token = re.compile(r"^(?:[A-Z]|Z[A-Z]?)$")
        canonical = re.compile(r"^[A-Z][A-Z0-9_]*$")
        prohibited_abbreviation = re.compile(r"(?:^|_)(?:FWD|REV)(?:_|$)")
        for turns, target_y in ((1, 9.0), (2, 9.0), (3, 9.0), (4, 12.0), (5, 13.0)):
            _, primary, cl1, cl2 = self.geometry(turns, target_y)
            coils = list(primary.coils) + [cl1, cl2]
            for coil in coils:
                via_labels = tuple(getattr(coil, "via_labels", ()))
                for label in tuple(coil.points) + via_labels:
                    with self.subTest(turns=turns, coil=coil.name, label=label):
                        self.assertRegex(label, canonical)
                        self.assertIsNone(legacy_token.fullmatch(label))
                        self.assertIsNone(prohibited_abbreviation.search(label))
                for label in via_labels:
                    self.assertTrue(
                        label.endswith("_VIA"),
                        f"{coil.name} via label {label} must end in _VIA",
                    )

    def test_primary_end_extension_automatically_clears_cl2_left_turnaround_vias(self):
        cfg = generator.build_config(
            {
                "number_of_secondary_turns": 5,
                "target_y_mm": 13.0,
                "primary_end_extension_mm": 0.0,
            }
        )
        dimensions = generator.calculate_dimensions(cfg)
        extension = (dimensions.primary_length_mm - dimensions.secondary_length_mm) / 2.0
        self.assertAlmostEqual(extension, 3.441413337162864, places=6)
        self.assertEqual(
            dimensions.primary_length_mm,
            dimensions.secondary_length_mm + (2.0 * extension),
        )

        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        self.assertIsNotNone(cl2)
        primary_segments = tuple(
            segment
            for coil in primary.coils
            for segment in coil.body_segments + coil.escape_segments
        )
        for via_label in cl2.via_labels:
            if via_label.endswith("_LEFT_TURNAROUND_VIA"):
                with self.subTest(via_label=via_label):
                    nearest_primary_trace = min(
                        generator.point_to_segment_distance(cl2.points[via_label], segment)
                        for segment in primary_segments
                    )
                    self.assertGreaterEqual(
                        nearest_primary_trace + generator.GEOMETRY_TOLERANCE_MM,
                        generator.osc1_via_trace_clearance(cfg),
                    )

    def test_primary_end_extension_accepts_safe_override_and_rejects_short_one(self):
        safe_cfg = generator.build_config(
            {
                "number_of_secondary_turns": 5,
                "target_y_mm": 13.0,
                "primary_end_extension_mm": 3.5,
            }
        )
        safe_dimensions = generator.calculate_dimensions(safe_cfg)
        self.assertAlmostEqual(
            (safe_dimensions.primary_length_mm - safe_dimensions.secondary_length_mm) / 2.0,
            3.5,
        )

        too_short_cfg = generator.build_config(
            {
                "number_of_secondary_turns": 5,
                "target_y_mm": 13.0,
                "primary_end_extension_mm": 3.0,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"set primary_end_extension_mm to 0 for the minimum viable setting",
        ):
            generator.calculate_dimensions(too_short_cfg)

    def test_primary_end_extension_uses_cl1_floor_without_cl2(self):
        one_turn_cfg = generator.build_config(
            {
                "number_of_secondary_turns": 1,
                "primary_end_extension_mm": 0.0,
            }
        )
        one_turn_dimensions = generator.calculate_dimensions(one_turn_cfg)
        one_turn_extension = (
            one_turn_dimensions.primary_length_mm
            - one_turn_dimensions.secondary_length_mm
        ) / 2.0
        self.assertGreater(
            one_turn_extension,
            one_turn_cfg["cl1_primary_end_min_clearance_mm"],
        )

        cfg = generator.build_config(
            {
                "generate_cl2": False,
                "primary_end_extension_mm": 0.0,
            }
        )
        dimensions = generator.calculate_dimensions(cfg)
        extension = (dimensions.primary_length_mm - dimensions.secondary_length_mm) / 2.0
        self.assertEqual(extension, cfg["cl1_primary_end_min_clearance_mm"])

    def test_generated_receiver_intermediate_pads_have_exactly_one_coil_prefix(self):
        cfg, _, cl1, cl2 = self.geometry(5, 13.0)
        external = {
            cfg["cl1_output_pad_name"],
            cfg["cl1_return_pad_name"],
            cfg["cl2_output_pad_name"],
            cfg["cl2_return_pad_name"],
        }
        footprint = generator.render_footprint(cfg)
        pads = re.findall(r'\(pad "([^"]+)" thru_hole', footprint)
        for name in pads:
            if name in external or name in {
                cfg["osc1_output_pad_name"],
                cfg["osc2_output_pad_name"],
                cfg["primary_input_pad_name"],
            }:
                continue
            with self.subTest(pad=name):
                self.assertRegex(name, r"^CL[12]_[A-Z][A-Z0-9_]*$")
                self.assertNotRegex(name, r"^CL[12]_CL[12]_" )

    def test_terminal_names_remain_the_configured_external_interface(self):
        cfg, _, _, _ = self.geometry(3, 9.0)
        footprint = generator.render_footprint(cfg)
        pads = set(re.findall(r'\(pad "([^"]+)" thru_hole', footprint))
        self.assertTrue(
            {
                cfg["osc1_output_pad_name"],
                cfg["osc2_output_pad_name"],
                cfg["primary_input_pad_name"],
                cfg["cl1_output_pad_name"],
                cfg["cl1_return_pad_name"],
                cfg["cl2_output_pad_name"],
                cfg["cl2_return_pad_name"],
            }.issubset(pads)
        )

    def test_strict_layout_validation_remains_enabled_for_boundary_turn_counts(self):
        for turns, target_y in ((1, 9.0), (5, 13.0)):
            with self.subTest(turns=turns):
                self.geometry(turns, target_y, validate=True)


if __name__ == "__main__":
    unittest.main()
