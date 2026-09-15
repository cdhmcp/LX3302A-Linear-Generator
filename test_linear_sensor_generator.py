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
    "2:9:left": "5e7bd7c7ca319c6fb3df9c7456872795e81abd9d415d37aa1895200cd85642de",
    "2:9:right": "1bd5e2f4fdbe6ddbb7ae0c7f6e40a46f990f8acbfe597fb17f3b6e3aa2c142e7",
    "3:9:left": "91b53292207e4ad202c3da66f7e287eef9d8ad2e91eb86e30ba239b4f4f83d59",
    "3:9:right": "6585164644adda4ba6247a847570f3f51eb74bfd6a6793ee91f23d364a6c21c4",
    "4:12:left": "a211f91577f8cf35fd71377e4e86af91ac2303cc9e9c9f0cd4c81a850a0c8812",
    "4:12:right": "51316dac57265f36a87e2fe35cbab5759ed36bba724789889b50eb9b1be1f39a",
    "5:13:left": "73510c51a5269ecb1f256d8adaa957114d3b89bbf0a613364ecb04111902442b",
    "5:13:right": "81085604acc327e2b08472903b98ea5981373fc39039fdc3f5b72fb4f7199b8e",
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
