"""Regression coverage for the point-driven linear sensor footprint generator."""

from __future__ import annotations

import hashlib
import json
import re
import unittest

import linear_sensor_generator as generator


FINGERPRINTS = {
    "1:9:left": "89580883348a793200a11ff24875bf5a075b8ca95fd124508007ea55e8be43ec",
    "1:9:right": "0742456b9839d40708e8606d7d1ce3547ad1dc003644a54f28ba7d0df7ccaf2a",
    "2:9:left": "b9f02254c6dee13e768e5297f6b016d98db30beb7a04599fcdd27d3d841f78f6",
    "2:9:right": "3de0e60d64c71ca48fa2fdd768a66630ba388dce20197a3071c36e092a853882",
    "3:9:left": "43a1ed11a7aa886eaff22af5ad4917fea084e38fcb903fa6ba475707c7095c68",
    "3:9:right": "1263bfb7bdeabc95f083158cb4f4bdf3186c6558d2bd56578571aa2b1efb3c41",
    "4:12:left": "0c7191ffa8432f41799827c4e1a8c6f1a5d3ac00b7473cfe5fc11275a1a284f8",
    "4:12:right": "19c8fc0961bec3ee82baee3f24b538bd305523604097612af62bad0daf651ecd",
    "5:13:left": "d460d6e8579fb70373ea2c17e958608d7440030c558abe71459bd352e9fdf6b8",
    "5:13:right": "8139035705885fb321eaf59f1f4541df2f3d019cf54da4b5ceab6f3159f19931",
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

    def test_cl2_left_turnaround_uses_on_rail_trim_and_compacts_five_turn_rack(self):
        cfg = generator.build_config(
            {
                "number_of_secondary_turns": 5,
                "target_y_mm": 13.0,
                "allow_invalid_geometry": True,
            }
        )
        dimensions = generator.calculate_dimensions(cfg)
        primary = generator.build_primary_geometry(cfg)
        layout = generator.build_multiturn_cl2_layout(cfg, dimensions, primary)
        half_span = generator.secondary_stroke_length(cfg) / 2.0
        via_pitch = generator.secondary_via_spacing(cfg)
        offsets = generator.secondary_turn_offsets(cfg)
        amplitude = generator.secondary_wave_amplitude_for_offsets(dimensions, offsets)

        for label, station_x in layout.left_handoff_station_x.items():
            turn = int(label.removeprefix("TURN").split("_", 1)[0])
            phase_sign = -1.0 if label.endswith("_START") else 1.0
            expected = generator.secondary_rail_point(
                cfg,
                dimensions,
                station_x,
                phase_sign,
                offsets[turn - 1],
                amplitude_override=amplitude,
            )
            self.assertEqual(layout.points[label], expected)
            self.assertGreaterEqual(station_x, half_span * -1.0)
            self.assertLessEqual(station_x, (-half_span) + via_pitch)

        rack = [
            layout.points[f"TURN{turn}_LEFT_TURNAROUND_VIA"]
            for turn in range(1, 5)
        ]
        rack_x = rack[0][0]
        self.assertTrue(all(point[0] == rack_x for point in rack))
        self.assertAlmostEqual(rack_x, -46.65071333716286, places=6)
        for first, second in zip(rack, rack[1:]):
            self.assertAlmostEqual(second[1] - first[1], via_pitch)

        generator.validate_multiturn_cl2_clearance(cfg, dimensions, primary, layout)

        for handoff_index, route in enumerate(layout.left_target_handoff_paths):
            self.assertEqual(route[0][0], layout.points[f"TURN{handoff_index + 2}_START"])
        for handoff_index, route in enumerate(layout.left_inner_handoff_paths):
            self.assertEqual(route[0][0], layout.points[f"TURN{handoff_index + 1}_LEFT_END"])

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
