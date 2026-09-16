"""Regression coverage for the point-driven linear sensor footprint generator."""

from __future__ import annotations

import hashlib
import json
import re
import unittest

import linear_sensor_generator as generator


FINGERPRINTS = {
    "1:9:left": "6fd84bacc71fd217aa424a11ffc1587a0130b31534284e0df12b5ef176d25a9e",
    "1:9:right": "46da77fb6216a12c156372eea4d188d47dabb4521ce069d658bbb2b859ae8a86",
    "2:9:left": "3fec412d3ae8f8dc6a0a1c6f14662c06ae83589ee1a31b8e860ec2d95fdd5ef0",
    "2:9:right": "cfe9fdad375b5ea34cb50c416cc2046c6f6a5eef4cdd1bf207f43104dd68622d",
    "3:9:left": "621ff5d13675a9e04484ef6555d755fd1ee9decb529724fa681169da9b368357",
    "3:9:right": "46bdbe0ee771a93cb7d442cfe198e5ee992fc3c86533c7d807d134e9c5c29b0a",
    "4:12:left": "cfda0cce79e77ea83feecaacbb8ed1a8c14f708251f9869eec81d197d1ae16a1",
    "4:12:right": "a3f4e172d3a19f9b3dedd113de4cfcb18c78d21214a326f88db37c5c983f34ad",
    "5:13:left": "5de64e80f885a912f0114a7d7b444a9976a5c58fbe278ca3a6de3c666cf5e112",
    "5:13:right": "f78ece57d4b00839c5824d862734b4aeb98636fef34650c0d7e35cf5effd3d58",
}

DEFAULT_AUTOMATIC_FOOTPRINT_SHA256 = (
    "68e036232b1c77f78c52ad55c95774fa7bf3eafaacc80c72d006ce47e47bcc90"
)


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

    def test_automatic_primary_corridor_inset_preserves_the_established_position(self):
        cfg = generator.build_config({"allow_invalid_geometry": False})
        dimensions = generator.calculate_dimensions(cfg)
        primary = generator.build_primary_geometry(cfg)
        osc1 = primary.coils[0].points
        osc2 = primary.coils[1].points

        historical_auto_y = (
            -(
                (dimensions.primary_width_mm / 2.0)
                - ((cfg["number_of_primary_turns"] - 1) * generator.trace_pitch(cfg))
            )
            + generator.osc1_via_trace_clearance(cfg)
            + cfg["trace_spacing_mm"]
        )
        self.assertEqual(cfg["osc1_vin_exit_offset_mm"], 0.0)
        self.assertAlmostEqual(
            generator.effective_osc1_vin_exit_y(cfg, dimensions),
            generator.automatic_osc1_vin_exit_y(cfg, dimensions),
        )
        self.assertAlmostEqual(osc1["SHARED_VIN_VIA"][1], historical_auto_y)
        self.assertAlmostEqual(osc1["VIN_FANOUT_JOG"][1], historical_auto_y)
        self.assertAlmostEqual(osc2["SHARED_VIN_VIA"][1], historical_auto_y)
        self.assertEqual(
            hashlib.sha256(generator.render_footprint(cfg).encode()).hexdigest(),
            DEFAULT_AUTOMATIC_FOOTPRINT_SHA256,
        )

    def test_manual_primary_corridor_inset_moves_dependent_primary_routes(self):
        for fanout_side in ("left", "right"):
            with self.subTest(fanout_side=fanout_side):
                automatic_cfg = generator.build_config(
                    {"fanout_side": fanout_side, "allow_invalid_geometry": False}
                )
                dimensions = generator.calculate_dimensions(automatic_cfg)
                manual_inset = (
                    generator.minimum_osc1_vin_exit_inset(automatic_cfg, dimensions) + 0.5
                )
                manual_cfg = generator.build_config(
                    {
                        "fanout_side": fanout_side,
                        "osc1_vin_exit_offset_mm": manual_inset,
                        "allow_invalid_geometry": False,
                    }
                )
                automatic = generator.build_primary_geometry(automatic_cfg)
                manual = generator.build_primary_geometry(manual_cfg)
                automatic_osc1 = automatic.coils[0].points
                manual_osc1 = manual.coils[0].points
                automatic_osc2 = automatic.coils[1].points
                manual_osc2 = manual.coils[1].points
                expected_y = generator.primary_top_copper_edge_y(dimensions=dimensions, cfg=manual_cfg) + manual_inset
                vertical_shift = expected_y - automatic_osc1["SHARED_VIN_VIA"][1]

                self.assertAlmostEqual(manual_osc1["SHARED_VIN_VIA"][1], expected_y)
                self.assertAlmostEqual(manual_osc1["VIN_FANOUT_JOG"][1], expected_y)
                self.assertAlmostEqual(
                    manual_osc1["ENTRY_WINDING_JOG"][1]
                    - automatic_osc1["ENTRY_WINDING_JOG"][1],
                    vertical_shift,
                )
                self.assertAlmostEqual(
                    manual_osc2["ENTRY_WINDING_JOG"][1]
                    - automatic_osc2["ENTRY_WINDING_JOG"][1],
                    vertical_shift,
                )
                self.assertEqual(
                    manual_osc1["TERMINAL_OUTPUT_VIA"],
                    automatic_osc1["TERMINAL_OUTPUT_VIA"],
                )

    def test_too_small_manual_primary_corridor_inset_is_forceable_geometry_error(self):
        relaxed_cfg = generator.build_config({"osc1_vin_exit_offset_mm": 1.0})
        dimensions = generator.calculate_dimensions(relaxed_cfg)
        minimum = generator.minimum_osc1_vin_exit_inset(relaxed_cfg, dimensions)
        self.assertGreater(minimum, relaxed_cfg["osc1_vin_exit_offset_mm"])
        self.assertIsNotNone(generator.build_primary_geometry(relaxed_cfg))

        strict_cfg = generator.build_config(
            {
                "osc1_vin_exit_offset_mm": relaxed_cfg["osc1_vin_exit_offset_mm"],
                "allow_invalid_geometry": False,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"osc1_vin_exit_offset_mm=.*minimum primary corridor top-edge inset.*set .* to 0",
        ):
            generator.build_primary_geometry(strict_cfg)

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
        self.assertAlmostEqual(extension, 3.475439339553418, places=6)
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

    def test_primary_end_extension_automatically_clears_cl1_without_cl2(self):
        cfg = generator.build_config(
            {
                "generate_cl2": False,
                "primary_end_extension_mm": 0.0,
            }
        )
        dimensions = generator.calculate_dimensions(cfg)
        extension = (dimensions.primary_length_mm - dimensions.secondary_length_mm) / 2.0
        self.assertGreater(extension, 0.0)

        primary = generator.build_primary_geometry(cfg)
        cl1 = generator.build_cl1_geometry(cfg, primary, None)
        self.assertIsNotNone(cl1)

        too_short_cfg = generator.build_config(
            {
                "generate_cl2": False,
                "primary_end_extension_mm": extension - 0.05,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"set primary_end_extension_mm to 0 for the minimum viable setting",
        ):
            generator.calculate_dimensions(too_short_cfg)

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
        # The one-turn boundary remains valid under strict checks.  The
        # current five-turn, 13 mm fixture is intentionally retained as a
        # renderable-but-invalid boundary case: relaxed generation supports
        # visual debugging while strict validation must still reject it.
        self.geometry(1, 9.0, validate=True)
        with self.assertRaisesRegex(
            ValueError,
            r"CL2 parallel sinusoidal traces violate configured spacing",
        ):
            self.geometry(5, 13.0, validate=True)


if __name__ == "__main__":
    unittest.main()
