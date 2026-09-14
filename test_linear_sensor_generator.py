import math
import unittest
from unittest import mock

import linear_sensor_generator as generator


class LinearSensorGeneratorTests(unittest.TestCase):
    def test_reference_dimensions_and_primary_layers(self) -> None:
        cfg = generator.build_config({"target_x_mm": 21.0, "target_y_mm": 7.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3})
        geometry = generator.build_primary_geometry(cfg)
        osc1, osc2 = geometry.coils

        self.assertEqual(geometry.dimensions.secondary_length_mm, 72.0)
        self.assertEqual(geometry.dimensions.secondary_width_mm, 5.5)
        self.assertEqual(geometry.dimensions.primary_length_mm, 78.0)
        self.assertEqual(geometry.dimensions.primary_width_mm, 5.65)
        self.assertEqual(osc1.name, "OSC1")
        self.assertEqual(osc1.layer, "B.Cu")
        self.assertEqual(osc1.escape_layer, "F.Cu")
        self.assertEqual(osc2.name, "OSC2")
        self.assertEqual(osc2.layer, "In2.Cu")
        self.assertEqual(osc2.escape_segments, ())

    def test_default_osc1_uses_annotated_three_turn_path(self) -> None:
        cfg = generator.build_config({"fanout_side": "left", "target_x_mm": 21.0, "target_y_mm": 9.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3})
        coil = generator.build_primary_geometry(cfg).coils[0]
        points = coil.points
        pitch = generator.trace_pitch(cfg)
        transition_shift = pitch * (math.sqrt(2.0) - 1.0)
        junction_separation = generator.parallel_45_junction_separation(cfg)

        self.assertEqual(coil.body_segments[0], (points["A"], points["A_JOG"]))
        self.assertEqual(points["A"][1], generator.terminal_row_y(cfg, "OSC1"))
        self.assertIn((points["A_JOG"], points["B"]), coil.body_segments)
        self.assertIn((points["B"], points["C"]), coil.body_segments)
        self.assertIn((points["H"], points["I"]), coil.body_segments)
        self.assertIn((points["N"], points["O"]), coil.body_segments)
        self.assertEqual(coil.body_segments[-1], (points["T"], points["U"]))
        self.assertEqual(
            coil.escape_segments,
            (
                (points["U"], points["VIN_JOG"]),
                (points["VIN_JOG"], points["VIN_APPROACH"]),
                (points["VIN_APPROACH"], points["V"]),
            ),
        )
        self.assertEqual(points["VIN_JOG"][1], points["U"][1])
        self.assertEqual(points["VIN_APPROACH"][1], points["V"][1])
        self.assertEqual(points["C"][1], points["B"][1])
        self.assertAlmostEqual(abs(points["I"][0] - points["H"][0]), pitch)
        self.assertAlmostEqual(abs(points["I"][1] - points["H"][1]), pitch)
        self.assertAlmostEqual(points["J"][0] - points["D"][0], pitch)
        self.assertAlmostEqual(points["P"][0] - points["J"][0], pitch)
        self.assertAlmostEqual(points["I"][1] - points["H"][1], pitch)
        self.assertAlmostEqual(points["O"][1] - points["N"][1], pitch)
        self.assertAlmostEqual(points["C"][1] - points["H"][1], junction_separation)
        self.assertAlmostEqual(points["N"][1] - points["H"][1], -transition_shift)
        self.assertGreaterEqual(
            generator.point_to_segment_distance(points["H"], (points["B"], points["C"])),
            pitch,
        )
        self.assertAlmostEqual(
            generator.point_to_segment_distance(points["N"], (points["H"], points["I"])),
            pitch,
        )

    def test_default_escape_coordinates_use_requested_terminal_boundary(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        points = generator.build_primary_geometry(cfg).coils[0].points
        dimensions = generator.calculate_dimensions(cfg)
        expected_terminal_x = -(dimensions.primary_length_mm / 2.0) - cfg["terminal_escape_length_mm"]

        self.assertEqual(points["A"][0], expected_terminal_x)
        self.assertEqual(points["V"][0], expected_terminal_x)
        self.assertEqual(points["A"][1], generator.terminal_row_y(cfg, "OSC1"))
        self.assertLess(points["U"][1], 0.0)
        self.assertEqual(points["VIN_JOG"][1], points["U"][1])
        self.assertEqual(points["V"][1], generator.terminal_row_y(cfg, "VIN"))

    def test_default_footprint_emits_oscillators_receivers_and_two_vin_vias(self) -> None:
        footprint = generator.render_footprint()

        self.assertIn('(footprint "LX3302A_LINEAR_SENSOR_COILS"', footprint)
        self.assertIn('(pad "OSC1" thru_hole', footprint)
        self.assertEqual(footprint.count('(pad "VIN" thru_hole'), 2)
        self.assertIn('(pad "OSC2" thru_hole', footprint)
        self.assertIn('(layer "B.Cu")', footprint)
        self.assertIn('(layer "F.Cu")', footprint)
        self.assertIn('(layer "In2.Cu")', footprint)
        self.assertIn('(pad "CL2" thru_hole', footprint)
        self.assertIn('(pad "CL2-GND" thru_hole', footprint)
        self.assertIn('(pad "CL1" thru_hole', footprint)
        self.assertIn('(pad "CL1-GND" thru_hole', footprint)
        self.assertEqual(footprint.count("(fp_arc "), 1)
        self.assertIn('(layer "In1.Cu")', footprint)

    def test_segment_distance_detects_crossing_touching_and_separated_segments(self) -> None:
        crossing_first = ((0.0, 0.0), (2.0, 2.0))
        crossing_second = ((0.0, 2.0), (2.0, 0.0))
        touching = ((2.0, 2.0), (3.0, 2.0))
        separated = ((0.0, 3.0), (2.0, 3.0))

        self.assertEqual(generator.segment_to_segment_distance(crossing_first, crossing_second), 0.0)
        self.assertEqual(generator.segment_to_segment_distance(crossing_first, touching), 0.0)
        self.assertGreater(generator.segment_to_segment_distance(crossing_first, separated), 0.0)

    def test_external_terminal_vias_share_compact_column_and_cl1_is_straight(self) -> None:
        cfg = generator.build_config()
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        assert cl1 is not None and cl2 is not None
        terminals = (
            ("OSC2", primary.pads["OSC2_A"]),
            ("VIN", primary.pads["VIN_V"]),
            ("OSC1", primary.pads["OSC1_A"]),
            ("CL2", cl2.points["A"]),
            ("CL2-GND", cl2.points["ZP"]),
            ("CL1-GND", cl1.points["ZN"]),
            ("CL1", cl1.points["A"]),
        )
        expected_x = generator.terminal_column_x(cfg, primary.dimensions)
        expected_spacing = generator.terminal_pad_pitch(cfg)

        for name, point in terminals:
            self.assertAlmostEqual(point[0], expected_x)
            self.assertAlmostEqual(point[1], generator.terminal_row_y(cfg, name))
        for (_, first), (_, second) in zip(terminals, terminals[1:]):
            self.assertAlmostEqual(generator.distance(first, second), expected_spacing)
        self.assertEqual(cl1.points["A"][1], cl1.points["A_FANOUT_JOG"][1])
        self.assertNotIn("C", cl1.points)
        self.assertIn(
            (cl1.points["B"], cl1.points["D_ENTRY_45_START"]),
            cl1.target_segments,
        )
        self.assertIn(
            (cl1.points["D_ENTRY_45_START"], cl1.points["D_ENTRY_45_END"]),
            cl1.target_segments,
        )
        self.assertIn(
            (cl1.points["D_ENTRY_45_END"], cl1.points["D"]),
            cl1.target_segments,
        )

    def test_bottom_target_mirrors_primary_and_escape_layers(self) -> None:
        cfg = generator.build_config({"target_side": "bottom"})
        osc1, osc2 = generator.build_primary_geometry(cfg).coils

        self.assertEqual(osc1.layer, "F.Cu")
        self.assertEqual(osc1.escape_layer, "B.Cu")
        self.assertEqual(osc2.layer, "In1.Cu")

    def test_right_fanout_mirrors_point_map_horizontally(self) -> None:
        left_geometry = generator.build_primary_geometry(generator.build_config({"fanout_side": "left"}))
        cfg = generator.build_config({"fanout_side": "right"})
        right_geometry = generator.build_primary_geometry(cfg)

        for name in ("A", "B", "C", "D", "E", "F", "U", "VIN_JOG", "V"):
            self.assertAlmostEqual(right_geometry.coils[0].points[name][0], -left_geometry.coils[0].points[name][0])
            self.assertAlmostEqual(right_geometry.coils[0].points[name][1], left_geometry.coils[0].points[name][1])
        for name in ("A", "A_JOG", "B", "C", "D", "E", "F", "X"):
            self.assertAlmostEqual(right_geometry.coils[1].points[name][0], -left_geometry.coils[1].points[name][0])
            self.assertAlmostEqual(right_geometry.coils[1].points[name][1], left_geometry.coils[1].points[name][1])

    def test_right_fanout_receiver_terminal_stubs_route_toward_sensor(self) -> None:
        cfg = generator.build_config({"fanout_side": "right"})
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        assert cl1 is not None and cl2 is not None

        sensor_edge_x = generator.secondary_stroke_length(cfg) / 2.0
        terminal_x = generator.terminal_column_x(cfg, primary.dimensions)

        self.assertEqual(cl1.points["A"][0], terminal_x)
        self.assertLess(cl1.points["B"][0], cl1.points["A"][0])
        self.assertGreater(cl1.points["B"][0], cl1.points["C"][0])
        self.assertGreater(cl1.points["C"][0], 0.0)
        self.assertEqual(cl1.points["A"][1], cl1.points["B"][1])
        self.assertEqual(cl1.points["B"][1], cl1.points["C"][1])
        self.assertEqual(cl1.points["ZN"][0], terminal_x)
        self.assertLess(cl1.points["ZM"][0], cl1.points["ZN"][0])
        self.assertEqual(cl1.points["ZM"][0], cl1.points["ZN_JOG"][0])
        self.assertEqual(cl1.points["ZN_JOG"][1], cl1.points["ZN"][1])

        self.assertEqual(cl2.points["A"][0], terminal_x)
        self.assertLess(cl2.points["B"][0], cl2.points["A"][0])
        self.assertGreater(cl2.points["B"][0], sensor_edge_x)
        self.assertEqual(cl2.points["A"][1], cl2.points["A_FANOUT_JOG"][1])
        self.assertEqual(cl2.points["A_FANOUT_JOG"][0], cl2.points["B"][0])
        self.assertEqual(cl2.points["ZP"][0], terminal_x)
        self.assertLess(cl2.points["ZO"][0], cl2.points["ZP"][0])
        self.assertGreater(cl2.points["ZO"][0], sensor_edge_x)
        self.assertEqual(cl2.points["ZO"][0], cl2.points["ZP_FANOUT_JOG"][0])
        self.assertEqual(cl2.points["ZP_FANOUT_JOG"][1], cl2.points["ZP"][1])

    def test_configurable_fourth_turn_gets_generated_labels(self) -> None:
        cfg = generator.build_config({"number_of_primary_turns": 4})
        osc1, osc2 = generator.build_primary_geometry(cfg).coils

        self.assertIn("TURN4_START", osc1.points)
        self.assertIn("OSC2_TURN4_START", osc2.points)
        self.assertEqual(osc1.body_segments[-1][1], osc1.points["U"])
        self.assertEqual(osc2.body_segments[-1][1], osc2.points["X"])

    def test_u_via_clearance_is_derived_from_via_and_trace_properties(self) -> None:
        cfg = generator.build_config({"fanout_side": "left", "target_x_mm": 21.0, "target_y_mm": 9.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3})
        points = generator.build_primary_geometry(cfg).coils[0].points
        expected_clearance = generator.osc1_via_trace_clearance(cfg)

        self.assertAlmostEqual(points["U"][0] - points["T"][0], expected_clearance)
        self.assertAlmostEqual(abs(points["U"][1] - points["T"][1]), expected_clearance)

    def test_vin_via_is_derived_from_inner_turn_clearance(self) -> None:
        cfg = generator.build_config()
        points = generator.build_primary_geometry(cfg).coils[0].points
        pitch = generator.trace_pitch(cfg)
        via_clearance = generator.osc1_via_trace_clearance(cfg)
        dimensions = generator.calculate_dimensions(cfg)
        inner_turn = cfg["number_of_primary_turns"] - 1
        inner_near_x = -(
            (dimensions.primary_length_mm / 2.0) - (inner_turn * pitch)
        )
        inner_top_y = -(
            (dimensions.primary_width_mm / 2.0) - (inner_turn * pitch)
        )

        self.assertAlmostEqual(points["U"][0], inner_near_x + via_clearance)
        self.assertAlmostEqual(
            points["U"][1], inner_top_y + via_clearance + cfg["trace_spacing_mm"]
        )
        self.assertEqual(points["V"][1], generator.terminal_row_y(cfg, "VIN"))
        self.assertEqual(points["V"][0], points["A"][0])

    def test_oscillator_fanout_routes_use_orthogonal_segments(self) -> None:
        osc1, osc2 = generator.build_primary_geometry().coils
        points1 = osc1.points
        points2 = osc2.points

        self.assertEqual(points1["A"][1], points1["A_JOG"][1])
        self.assertEqual(points1["A_JOG"][0], points1["B"][0])
        self.assertEqual(points1["B"][1], points1["C"][1])
        self.assertEqual(points1["U"][1], points1["VIN_JOG"][1])
        self.assertEqual(points1["VIN_JOG"][0], points1["VIN_APPROACH"][0])
        self.assertEqual(points1["VIN_APPROACH"][1], points1["V"][1])

        self.assertEqual(points2["A"][1], points2["A_JOG"][1])
        self.assertEqual(points2["A_JOG"][0], points2["B"][0])
        self.assertEqual(points2["B"][0], points2["C"][0])
        self.assertEqual(points2["C"][1], points2["F"][1])
        self.assertNotEqual(points2["F"][1], 0.0)
        terminal_x = points2["A"][0]
        self.assertTrue(
            all(
                point[0] >= terminal_x
                for point in (points2["A"], points2["A_JOG"], points2["B"], points2["C"])
            )
        )

    def test_moved_vin_via_clears_cl2_left_turnaround_for_all_turn_counts(self) -> None:
        for fanout_side in ("left", "right"):
            for turns in range(1, 6):
                with self.subTest(fanout_side=fanout_side, turns=turns):
                    cfg = generator.build_config(
                        {
                            "fanout_side": fanout_side,
                            "number_of_secondary_turns": turns,
                            "allow_invalid_geometry": False,
                        }
                    )
                    primary = generator.build_primary_geometry(cfg)
                    cl2 = generator.build_cl2_geometry(cfg, primary)
                    assert cl2 is not None
                    vin = primary.pads["VIN_U"]
                    detours = tuple(
                        point
                        for label, point in cl2.points.items()
                        if label.endswith("_LEFT_DETOUR_VIA")
                    )
                    for detour in detours:
                        self.assertGreaterEqual(
                            generator.distance(vin, detour) + 1e-9,
                            generator.terminal_pad_pitch(cfg),
                        )

    def test_transition_vias_follow_the_tighter_receiver_or_primary_envelope(self) -> None:
        cfg = generator.build_config({"number_of_secondary_turns": 5, "target_y_mm": 13.0})
        dimensions = generator.calculate_dimensions(cfg)
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        assert cl2 is not None and cl1 is not None

        clearance = generator.osc1_via_trace_clearance(cfg)
        primary_upper = -generator.primary_inner_half_height(cfg, dimensions) + clearance
        for coil, labels in (
            (cl2, tuple(f"TURN{turn}_LEFT_UPPER_VIA" for turn in range(1, 6))),
            (cl1, tuple(f"TURN{turn}_FWD_MID_VIA" for turn in range(1, 6))),
        ):
            upper = tuple(coil.points[label][1] for label in labels)
            self.assertAlmostEqual(upper[0], upper[4], places=4)
            self.assertAlmostEqual(upper[1], upper[3], places=4)
            # The middle transition has no local receiver obstacle, so it
            # remains at the primary-derived boundary.  Its neighbors move
            # only enough to clear active local rail spans, not full cycles.
            self.assertAlmostEqual(upper[2], primary_upper, places=4)
            self.assertLess(upper[1], upper[0])
            self.assertLess(upper[2], upper[1])

        self.assertAlmostEqual(
            cl2.points["TURN1_RIGHT_LOWER_VIA"][1],
            -cl2.points["TURN1_LEFT_UPPER_VIA"][1],
            places=4,
        )

    def test_receiver_ground_escape_rows_stay_fixed_when_terminal_order_changes(self) -> None:
        cfg = generator.build_config()
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        assert cl1 is not None and cl2 is not None

        self.assertEqual(cl1.points["ZM"][1], generator.terminal_row_y(cfg, "CL1"))
        self.assertEqual(cl1.points["ZN"][1], generator.terminal_row_y(cfg, "CL1-GND"))
        self.assertEqual(cl1.points["ZM"][0], cl1.points["ZN_JOG"][0])
        self.assertEqual(cl1.points["ZN_JOG"][1], cl1.points["ZN"][1])
        self.assertEqual(cl2.points["ZO"][1], 0.0)
        self.assertEqual(cl2.points["ZP"][1], generator.terminal_row_y(cfg, "CL2-GND"))

    def test_invalid_secondary_width_is_rejected(self) -> None:
        cfg = generator.build_config()
        cfg["secondary_y_reduction_mm"] = cfg["target_y_mm"]

        with self.assertRaisesRegex(ValueError, "positive secondary width"):
            generator.build_primary_geometry(cfg)

    def test_excessive_turn_count_is_rejected(self) -> None:
        cfg = generator.build_config({"number_of_primary_turns": 30})

        with self.assertRaisesRegex(ValueError, "Primary width is insufficient"):
            generator.build_primary_geometry(cfg)

    def test_default_osc2_reverses_overlaid_perimeters_and_shares_vin_via(self) -> None:
        osc1, osc2 = generator.build_primary_geometry().coils
        points = osc2.points

        self.assertEqual(points["X"], osc1.points["U"])
        self.assertEqual(points["G"], osc1.points["G"])
        self.assertEqual(points["J"], osc1.points["D"])
        self.assertEqual(points["M"], osc1.points["M"])
        self.assertEqual(points["P"], osc1.points["J"])
        self.assertEqual(points["S"], osc1.points["S"])
        self.assertEqual(points["V"], osc1.points["P"])
        self.assertEqual(points["A"][1], points["A_JOG"][1])
        self.assertEqual(points["B"][0], points["A_JOG"][0])
        self.assertEqual(points["B"][1], points["F"][1])
        self.assertIn((osc1.points["G"], osc1.points["F"]), osc2.body_segments)
        self.assertIn((osc1.points["F"], osc1.points["E"]), osc2.body_segments)
        self.assertIn((osc1.points["M"], osc1.points["L"]), osc2.body_segments)
        self.assertIn((osc1.points["S"], osc1.points["R"]), osc2.body_segments)
        self.assertEqual(osc2.body_segments[-1], (points["W"], points["X"]))

    def test_osc2_can_be_disabled_independently(self) -> None:
        cfg = generator.build_config({"generate_osc2": False})

        self.assertEqual([coil.name for coil in generator.build_primary_geometry(cfg).coils], ["OSC1"])
        self.assertNotIn('(pad "OSC2" thru_hole', generator.render_footprint(cfg))

    def test_osc2_requires_osc1_shared_vin_transition(self) -> None:
        cfg = generator.build_config({"generate_osc1": False, "generate_osc2": True})

        with self.assertRaisesRegex(ValueError, "OSC2 requires OSC1"):
            generator.build_primary_geometry(cfg)

    def test_reference_cl2_span_layers_and_outer_extrema(self) -> None:
        cfg = generator.build_config({"target_x_mm": 21.0, "target_y_mm": 7.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3, "fanout_side": "left"})
        cl2 = generator.build_cl2_geometry(cfg)
        self.assertIsNotNone(cl2)
        assert cl2 is not None

        self.assertEqual(cl2.target_layer, "F.Cu")
        self.assertEqual(cl2.inner_layer, "In1.Cu")
        self.assertEqual(cl2.stroke_length_mm, 71.0)
        self.assertEqual(cl2.points["C"], (-35.5, 0.0))
        self.assertEqual(cl2.points["J"][0], 35.5)
        self.assertEqual(cl2.points["ZN"], cl2.points["C"])
        self.assertEqual(cl2.points["D"][1], -2.75)
        self.assertEqual(cl2.points["G"][1], 2.75)

    def test_adjusted_target_height_generates_receiver_geometry(self) -> None:
        for target_y_mm in (7.5, 9.0):
            with self.subTest(target_y_mm=target_y_mm):
                cfg = generator.build_config({"target_y_mm": target_y_mm})
                footprint = generator.render_footprint(cfg)

                self.assertIn('(pad "CL1" thru_hole', footprint)
                self.assertIn('(pad "CL2" thru_hole', footprint)

    def test_excessive_target_height_reports_receiver_spacing_failure(self) -> None:
        cfg = generator.build_config({"target_y_mm": 30.0})

        with self.assertRaisesRegex(ValueError, "parallel sinusoidal traces"):
            generator.render_footprint(cfg)

    def test_allow_invalid_geometry_skips_receiver_validation_for_debug_rendering(self) -> None:
        strict_cfg = generator.build_config({"number_of_secondary_turns": 3})
        with mock.patch.object(
            generator,
            "validate_multiturn_cl1_clearance",
            side_effect=ValueError("debug receiver violation"),
        ):
            with self.assertRaisesRegex(ValueError, "debug receiver violation"):
                generator.build_cl1_geometry(strict_cfg)

        debug_cfg = generator.build_config(
            {"number_of_secondary_turns": 3, "allow_invalid_geometry": True}
        )
        with mock.patch.object(
            generator,
            "validate_multiturn_cl1_clearance",
            side_effect=ValueError("debug receiver violation"),
        ):
            cl1 = generator.build_cl1_geometry(debug_cfg)

        self.assertIsNotNone(cl1)

    def test_multiturn_receivers_build_in_strict_mode_when_single_rack_fits(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)

                self.assertIsNotNone(cl2)
                self.assertIsNotNone(cl1)

    def test_multiturn_cl1_reports_when_compact_height_cannot_fit_single_rack(self) -> None:
        for turns, target_y_mm in ((4, 9.0), (5, 9.0), (5, 12.0)):
            with self.subTest(turns=turns, target_y_mm=target_y_mm):
                cfg = generator.build_config(
                    {"number_of_secondary_turns": turns, "target_y_mm": target_y_mm}
                )
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                with self.assertRaisesRegex(
                    ValueError,
                    "right lower-via rack cannot fit",
                ):
                    generator.build_cl1_geometry(cfg, primary, cl2)

    def test_two_turn_receivers_use_generalized_builders_and_keep_point_aliases(self) -> None:
        cfg = generator.build_config({"number_of_secondary_turns": 2})
        primary = generator.build_primary_geometry(cfg)

        with mock.patch.object(
            generator, "build_cl2_point_map", side_effect=AssertionError("legacy CL2 point map used")
        ), mock.patch.object(
            generator, "build_cl2_segments", side_effect=AssertionError("legacy CL2 segments used")
        ), mock.patch.object(
            generator,
            "validate_cl2_clearance",
            side_effect=AssertionError("legacy CL2 validation used"),
        ), mock.patch.object(
            generator, "build_cl1_point_map", side_effect=AssertionError("legacy CL1 point map used")
        ), mock.patch.object(
            generator, "build_cl1_routes", side_effect=AssertionError("legacy CL1 routes used")
        ), mock.patch.object(
            generator,
            "validate_cl1_clearance",
            side_effect=AssertionError("legacy CL1 validation used"),
        ):
            cl2 = generator.build_cl2_geometry(cfg, primary)
            cl1 = generator.build_cl1_geometry(cfg, primary, cl2)

        assert cl2 is not None and cl1 is not None
        for label in ("C", "J", "T", "U", "ZE", "ZN", "ZP"):
            self.assertIn(label, cl2.points)
        for label in ("E", "K", "L", "T", "U", "ZB", "ZC", "ZJ", "ZN"):
            self.assertIn(label, cl1.points)
        self.assertNotIn("C", cl1.points)
        self.assertEqual(cl2.via_labels, generator.CL2_TWO_TURN_LEGACY_VIA_LABELS)
        self.assertIn("TURN1_LEFT_UPPER_VIA", cl1.via_labels)
        self.assertIn("TURN2_LEFT_LOWER_VIA", cl1.via_labels)

    def test_multiturn_cl2_columns_center_on_quarter_span_and_stay_inside_outer_envelope(self) -> None:
        for turns in (1, 3, 4, 5):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                dimensions = generator.calculate_dimensions(cfg)
                cl2 = generator.build_cl2_geometry(cfg)
                assert cl2 is not None
                half_span = generator.secondary_stroke_length(cfg) / 2.0
                quarter_span = half_span / 2.0
                via_spacing = generator.secondary_via_spacing(cfg)
                midpoint = (turns - 1) / 2.0
                expected_left_columns = [
                    -quarter_span + ((midpoint - index) * via_spacing)
                    for index in range(turns)
                ]
                actual_left_columns = [
                    cl2.points[f"TURN{index + 1}_LEFT_OUTER"][0]
                    for index in range(turns)
                ]

                for actual, expected in zip(actual_left_columns, expected_left_columns):
                    self.assertAlmostEqual(actual, expected)
                self.assertLessEqual(
                    max(abs(point[1]) for point in cl2.points.values()),
                    (dimensions.secondary_width_mm / 2.0) + 0.01,
                )

    def test_multiturn_cl2_right_end_turnaround_shares_the_forward_curve_endpoint(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                cfg = generator.build_config(
                    {"number_of_secondary_turns": turns, "allow_invalid_geometry": False}
                )
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                assert cl2 is not None

                half_span = generator.secondary_stroke_length(cfg) / 2.0
                quarter_span = half_span / 2.0
                outer_offsets = generator.secondary_turn_offsets(cfg)
                quarter_shifts = generator.cl2_quarter_column_shifts(cfg)
                amplitude_override = generator.secondary_wave_amplitude_for_offsets(
                    dimensions,
                    outer_offsets,
                )
                upper_via_y = -(
                    generator.primary_inner_half_height(cfg, dimensions)
                    - generator.osc1_via_trace_clearance(cfg)
                )
                lower_via_y = -upper_via_y

                for turn_index, outer_offset in enumerate(outer_offsets):
                    turn_number = turn_index + 1
                    shift = quarter_shifts[turn_index]
                    right_column_x = quarter_span + shift
                    reverse_right_column_x = quarter_span - shift
                    expected_right_end = generator.point_at_station_x(
                        generator.secondary_rail_point(
                            cfg,
                            dimensions,
                            half_span,
                            -1.0,
                            outer_offset,
                            amplitude_override=amplitude_override,
                        ),
                        half_span,
                    )

                    self.assertEqual(
                        cl2.points[f"TURN{turn_number}_RIGHT_END"],
                        expected_right_end,
                    )
                    self.assertEqual(
                        cl2.points[f"TURN{turn_number}_RIGHT_RUNUP"],
                        expected_right_end,
                    )
                    self.assertEqual(
                        cl2.points[f"TURN{turn_number}_RIGHT_LOWER_VIA"],
                        (right_column_x, lower_via_y),
                    )
                    self.assertEqual(
                        cl2.points[f"TURN{turn_number}_REV_RIGHT_UPPER_VIA"],
                        (reverse_right_column_x, upper_via_y),
                    )
                    self.assertIn(
                        (
                            cl2.points[f"TURN{turn_number}_RIGHT_END"],
                            cl2.points[f"TURN{turn_number}_RIGHT_DETOUR_VIA"],
                        ),
                        cl2.target_segments,
                    )
                    self.assertIn(
                        (
                            cl2.points[f"TURN{turn_number}_RIGHT_DETOUR_VIA"],
                            cl2.points[f"TURN{turn_number}_RIGHT_RUNUP"],
                        ),
                        cl2.inner_segments,
                    )
                    self.assertTrue(
                        any(
                            segment[0] == cl2.points[f"TURN{turn_number}_RIGHT_END"]
                            and segment[1] != cl2.points[f"TURN{turn_number}_RIGHT_DETOUR_VIA"]
                            for segment in cl2.inner_segments
                        )
                    )

                self.assertEqual(
                    len(
                        [
                            label
                            for label in (
                                f"TURN{turn_number}_RIGHT_DETOUR_VIA"
                                for turn_number in range(1, turns + 1)
                            )
                            if label in cl2.points
                        ]
                    ),
                    turns,
                )

    def test_multiturn_cl2_right_end_turnaround_packs_one_rightmost_centered_column(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                cfg = generator.build_config(
                    {"number_of_secondary_turns": turns, "allow_invalid_geometry": False}
                )
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                assert cl2 is not None

                detours = [
                    cl2.points[f"TURN{turn_number}_RIGHT_DETOUR_VIA"]
                    for turn_number in range(1, turns + 1)
                ]
                unique_x = {round(point[0], 6) for point in detours}
                expected_rightmost_u = (
                    (generator.secondary_stroke_length(cfg) / 2.0)
                    + generator.secondary_via_spacing(cfg)
                )
                expected_ys = generator.centered_positions(
                    turns,
                    generator.secondary_via_spacing(cfg),
                )

                self.assertEqual(len(unique_x), 1)
                self.assertAlmostEqual(
                    (-generator.fanout_direction(cfg)) * detours[0][0],
                    expected_rightmost_u,
                )
                for detour, expected_y in zip(detours, expected_ys):
                    self.assertAlmostEqual(detour[1], expected_y)
                for first, second in zip(detours, detours[1:]):
                    self.assertAlmostEqual(
                        generator.distance(first, second),
                        generator.secondary_via_spacing(cfg),
                    )

    def test_multiturn_cl2_left_end_turnaround_preserves_staggered_rail_anchors(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                cfg = generator.build_config(
                    {"number_of_secondary_turns": turns, "allow_invalid_geometry": True}
                )
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                assert cl2 is not None

                half_span = generator.secondary_stroke_length(cfg) / 2.0
                outer_offsets = generator.secondary_turn_offsets(cfg)
                amplitude_override = generator.secondary_wave_amplitude_for_offsets(
                    dimensions,
                    outer_offsets,
                )

                for turn_index, outer_offset in enumerate(outer_offsets):
                    turn_number = turn_index + 1
                    expected_start = generator.secondary_rail_point(
                        cfg,
                        dimensions,
                        -half_span,
                        -1.0,
                        outer_offset,
                        amplitude_override=amplitude_override,
                    )
                    expected_left_end = generator.secondary_rail_point(
                        cfg,
                        dimensions,
                        -half_span,
                        1.0,
                        outer_offset,
                        amplitude_override=amplitude_override,
                    )

                    self.assertEqual(cl2.points[f"TURN{turn_number}_START"], expected_start)
                    self.assertEqual(cl2.points[f"TURN{turn_number}_LEFT_END"], expected_left_end)

                    if turn_number < turns:
                        self.assertTrue(
                            any(
                                segment[0] == cl2.points[f"TURN{turn_number}_LEFT_END"]
                                for segment in cl2.inner_segments
                            )
                        )
                        self.assertTrue(
                            any(
                                segment[1] == cl2.points[f"TURN{turn_number + 1}_START"]
                                for segment in cl2.target_segments
                            )
                        )
                        self.assertTrue(
                            any(
                                segment[1] == cl2.points[f"TURN{turn_number}_LEFT_END"]
                                and segment[0] != cl2.points[f"TURN{turn_number}_LEFT_DETOUR_VIA"]
                                for segment in cl2.inner_segments
                            )
                        )

                self.assertEqual(
                    cl2.points[f"TURN{turns}_RETURN_START"],
                    cl2.points[f"TURN{turns}_LEFT_END"],
                )
                self.assertEqual(
                    len(
                        [
                            label
                            for label in (
                                f"TURN{turn_number}_LEFT_DETOUR_VIA"
                                for turn_number in range(1, turns)
                            )
                            if label in cl2.points
                        ]
                    ),
                    max(turns - 1, 0),
                )
                starts = [cl2.points[f"TURN{turn_number}_START"] for turn_number in range(1, turns + 1)]
                ends = [cl2.points[f"TURN{turn_number}_LEFT_END"] for turn_number in range(1, turns + 1)]
                for first, second in zip(starts, starts[1:]):
                    self.assertGreaterEqual(
                        generator.distance(first, second) + 1e-9,
                        generator.trace_pitch(cfg),
                    )
                for first, second in zip(ends, ends[1:]):
                    self.assertGreaterEqual(
                        generator.distance(first, second) + 1e-9,
                        generator.trace_pitch(cfg),
                    )

    def test_multiturn_cl2_left_end_turnaround_selects_a_compact_clearance_rack(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                cfg = generator.build_config(
                    {"number_of_secondary_turns": turns, "allow_invalid_geometry": True}
                )
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                assert cl2 is not None

                detours = [
                    cl2.points[f"TURN{turn_number}_LEFT_DETOUR_VIA"]
                    for turn_number in range(1, turns)
                ]

                if turns == 1:
                    self.assertEqual(detours, [])
                    continue

                edge_x = min(
                    *(
                        cl2.points[f"TURN{turn_number}_START"][0]
                        for turn_number in range(2, turns + 1)
                    ),
                    *(
                        cl2.points[f"TURN{turn_number}_LEFT_END"][0]
                        for turn_number in range(1, turns)
                    ),
                )
                via_spacing = generator.secondary_via_spacing(cfg)
                old_rack_x = edge_x - (4.0 * via_spacing)
                self.assertGreater(
                    sum(point[0] for point in detours) / len(detours),
                    old_rack_x + (0.5 * via_spacing),
                )
                for first, second in zip(detours, detours[1:]):
                    self.assertGreaterEqual(
                        generator.distance(first, second) + 1e-9,
                        via_spacing,
                    )

    def test_multiturn_cl2_left_handoff_bundle_keeps_trace_and_via_clearance(self) -> None:
        for turns in range(2, 6):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                layout = generator.build_multiturn_cl2_layout(cfg, dimensions, primary)
                pitch = generator.trace_pitch(cfg)

                for routes in (
                    layout.left_target_handoff_paths,
                    layout.left_inner_handoff_paths,
                ):
                    for first, second in zip(routes, routes[1:]):
                        self.assertGreaterEqual(
                            generator.path_to_path_distance(first, second) + 0.003,
                            pitch,
                        )

                for handoff_index, route in enumerate(layout.left_target_handoff_paths):
                    for path_index, path in enumerate(layout.target_forward_paths):
                        if path_index == handoff_index + 1:
                            continue
                        self.assertGreaterEqual(
                            generator.path_to_path_distance(route, path) + 0.003,
                            pitch,
                        )
                for handoff_index, route in enumerate(layout.left_inner_handoff_paths):
                    for path_index, path in enumerate(layout.inner_reverse_paths):
                        if path_index == handoff_index:
                            continue
                        self.assertGreaterEqual(
                            generator.path_to_path_distance(route, path) + 0.003,
                            pitch,
                        )

    def test_multiturn_cl2_fanout_escapes_share_y_zero_and_clear_turnarounds(self) -> None:
        for fanout_side in ("left", "right"):
            for turns in range(1, 6):
                with self.subTest(fanout_side=fanout_side, turns=turns):
                    cfg = generator.build_config(
                        {
                            "fanout_side": fanout_side,
                            "number_of_secondary_turns": turns,
                            "allow_invalid_geometry": False,
                        }
                    )
                    dimensions = generator.calculate_dimensions(cfg)
                    primary = generator.build_primary_geometry(cfg)
                    layout = generator.build_multiturn_cl2_layout(cfg, dimensions, primary)
                    pitch = generator.trace_pitch(cfg)
                    via_clearance = generator.via_to_trace_clearance(cfg)

                    self.assertEqual(layout.points["B"], layout.points["ZO"])
                    self.assertEqual(layout.points["B"][1], 0.0)
                    convergence = layout.points["CL2_FANOUT_CONVERGENCE"]
                    self.assertEqual(convergence[1], 0.0)
                    self.assertEqual(layout.entry_escape_path[0][0], layout.points["A"])
                    self.assertEqual(
                        layout.entry_escape_path[-1][1], layout.points["TURN1_START"]
                    )
                    self.assertEqual(
                        layout.return_escape_path[0][0],
                        layout.points[f"TURN{turns}_RETURN_START"],
                    )
                    self.assertEqual(layout.return_escape_path[-1][1], layout.points["ZP"])
                    if turns > 1:
                        self.assertGreaterEqual(len(layout.entry_escape_path), 3)
                        self.assertGreaterEqual(len(layout.return_escape_path), 3)
                        detours = tuple(
                            layout.points[f"TURN{turn_number}_LEFT_DETOUR_VIA"]
                            for turn_number in range(1, turns)
                        )
                        side = generator.fanout_direction(cfg)
                        outer_stack_x = (
                            min(point[0] for point in detours)
                            if side < 0.0
                            else max(point[0] for point in detours)
                        )
                        self.assertAlmostEqual(
                            convergence[0],
                            outer_stack_x
                            + (side * generator.via_to_trace_clearance(cfg)),
                        )
                        self.assertIn(
                            (layout.points["B"], convergence),
                            layout.entry_escape_path,
                        )
                        self.assertIn(
                            (convergence, layout.points["ZO"]),
                            layout.return_escape_path,
                        )
                        target_escape_start = layout.entry_escape_path[
                            layout.entry_escape_path.index((layout.points["B"], convergence)) + 1
                        ]
                        return_escape_end = layout.return_escape_path[
                            layout.return_escape_path.index((convergence, layout.points["ZO"])) - 1
                        ]
                        self.assertLess(target_escape_start[1][1], 0.0)
                        self.assertGreater(return_escape_end[0][1], 0.0)

                    checks = (
                        (
                            layout.entry_escape_path,
                            layout.target_segments,
                            layout.target_forward_paths[0],
                        ),
                        (
                            layout.return_escape_path,
                            layout.inner_segments,
                            layout.inner_reverse_paths[-1],
                        ),
                    )
                    for escape, layer_segments, connected_path in checks:
                        obstacles = tuple(
                            segment
                            for segment in layer_segments
                            if segment not in escape and segment not in connected_path
                        )
                        for segment in escape:
                            self.assertTrue(
                                all(
                                    generator.segment_to_segment_distance(segment, obstacle)
                                    + generator.ROUTING_POLYGONAL_TOLERANCE_MM
                                    >= pitch
                                    for obstacle in obstacles
                                )
                            )
                            self.assertTrue(
                                all(
                                    generator.point_to_segment_distance(
                                        layout.points[via_label], segment
                                    )
                                    + generator.ROUTING_POLYGONAL_TOLERANCE_MM
                                    >= via_clearance
                                    for via_label in layout.via_labels
                                    if via_label not in ("A", "ZP")
                                )
                            )

    def test_multiturn_cl1_columns_center_on_midpoint_and_step_inward(self) -> None:
        for turns in (1, 3, 4, 5):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None
                via_spacing = generator.secondary_via_spacing(cfg)
                half_span = generator.secondary_stroke_length(cfg) / 2.0
                expected_midpoints = list(generator.cl1_midpoint_columns(cfg))
                actual_midpoints = [
                    cl1.points[f"TURN{index + 1}_FWD_MID_END"][0]
                    for index in range(turns)
                ]
                actual_reverse_midpoints = [
                    cl1.points[f"TURN{index + 1}_REV_MID_END"][0]
                    for index in range(turns)
                ]
                expected_right_columns = [
                    half_span - (index * via_spacing)
                    for index in range(turns)
                ]
                actual_right_columns = [
                    cl1.points[f"TURN{index + 1}_RIGHT_UPPER_VIA"][0]
                    for index in range(turns)
                ]

                for actual, expected in zip(actual_midpoints, expected_midpoints):
                    self.assertAlmostEqual(actual, expected)
                for forward, reverse in zip(actual_midpoints, actual_reverse_midpoints):
                    self.assertAlmostEqual(reverse, -forward)
                for actual, expected in zip(actual_right_columns, expected_right_columns):
                    self.assertAlmostEqual(actual, expected)
                expected_left_columns = [
                    -half_span + (index * via_spacing)
                    for index in range(turns)
                ]
                actual_left_lower_columns = [
                    cl1.points[
                        "D" if index == 0 else f"TURN{index + 1}_LEFT_LOWER_VIA"
                    ][0]
                    for index in range(turns)
                ]
                actual_left_upper_columns = [
                    cl1.points[f"TURN{index + 1}_LEFT_UPPER_VIA"][0]
                    for index in range(turns)
                ]
                for actual, expected in zip(actual_left_lower_columns, expected_left_columns):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(actual_left_upper_columns, [-half_span] * turns)
                self.assertLessEqual(
                    max(abs(point[1]) for point in cl1.points.values()),
                    (dimensions.secondary_width_mm / 2.0) + 0.01,
                )

    def test_cl1_crossover_candidate_cl2_segments_only_include_local_x_overlaps(self) -> None:
        for fanout_side in ("left", "right"):
            with self.subTest(fanout_side=fanout_side):
                cfg = generator.build_config(
                    {
                        "number_of_secondary_turns": 5,
                        "fanout_side": fanout_side,
                        "secondary_curve_samples_per_cycle": 64,
                    }
                )
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                assert cl2 is not None

                clearance = generator.osc1_via_trace_clearance(cfg)
                all_segments = cl2.target_segments + cl2.inner_segments
                if generator.fanout_direction(cfg) > 0:
                    all_segments = generator.mirror_segments_horizontally(all_segments)

                for turn_index in range(cfg["number_of_secondary_turns"]):
                    turn_x = generator.cl1_right_end_column(cfg, turn_index)
                    minimum_x = turn_x - clearance - generator.GEOMETRY_TOLERANCE_MM
                    maximum_x = turn_x + clearance + generator.GEOMETRY_TOLERANCE_MM
                    expected = tuple(
                        segment
                        for segment in all_segments
                        if min(segment[0][0], segment[1][0]) <= maximum_x
                        and max(segment[0][0], segment[1][0]) >= minimum_x
                    )

                    actual = generator.cl1_crossover_candidate_cl2_segments(
                        cfg, cl2, turn_x, clearance
                    )

                    self.assertEqual(actual, expected)
                    self.assertGreater(len(actual), 0)
                    self.assertLess(len(actual), len(all_segments))

    def test_multiturn_cl1_left_transition_handoff_uses_flipped_via_groups(self) -> None:
        for turns in (3, 5):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None

                self.assertEqual(cl1.points["TURN1_LEFT_LOWER_VIA"], cl1.points["D"])
                for turn_index in range(turns - 1):
                    turn = turn_index + 1
                    next_lower_via = cl1.points[f"TURN{turn + 1}_LEFT_LOWER_VIA"]
                    upper_via = cl1.points[f"TURN{turn}_LEFT_UPPER_VIA"]
                    crossover_jog = cl1.points[f"TURN{turn}_LEFT_CROSSOVER_JOG"]
                    next_start = cl1.points[f"TURN{turn + 1}_START"]
                    self.assertEqual(crossover_jog, (next_lower_via[0], upper_via[1]))
                    if upper_via != crossover_jog:
                        self.assertIn((upper_via, crossover_jog), cl1.crossover_segments)
                    self.assertIn((crossover_jog, next_lower_via), cl1.crossover_segments)
                    self.assertIn((next_lower_via, next_start), cl1.target_segments)

                final_upper_via = cl1.points[f"TURN{turns}_LEFT_UPPER_VIA"]
                final_escape_jog = cl1.points["LEFT_RETURN_ESCAPE_JOG"]
                final_escape_via = cl1.points["LEFT_RETURN_ESCAPE_VIA"]
                self.assertLess(final_escape_via[0], final_upper_via[0])
                self.assertIn((final_upper_via, final_escape_jog), cl1.crossover_segments)
                self.assertIn((final_escape_jog, final_escape_via), cl1.crossover_segments)
                self.assertAlmostEqual(
                    cl1.points["B"][1],
                    final_escape_via[1],
                )
                entry_45_start = cl1.points["D_ENTRY_45_START"]
                entry_45_end = cl1.points["D_ENTRY_45_END"]
                self.assertAlmostEqual(
                    abs(entry_45_end[0] - entry_45_start[0]),
                    abs(entry_45_end[1] - entry_45_start[1]),
                )
                self.assertIn((cl1.points["B"], entry_45_start), cl1.target_segments)
                self.assertIn((entry_45_start, entry_45_end), cl1.target_segments)
                self.assertIn((entry_45_end, cl1.points["D"]), cl1.target_segments)
                self.assertIn(
                    (final_escape_via, cl1.points["LEFT_RETURN_FANOUT_JOG"]),
                    cl1.inner_segments,
                )
                self.assertEqual(
                    cl1.points["LEFT_RETURN_FANOUT_JOG"][1], final_escape_via[1]
                )

    def test_receiver_terminal_rows_place_cl2_above_cl1(self) -> None:
        cfg = generator.build_config()
        self.assertLess(
            generator.terminal_row_y(cfg, "CL2"),
            generator.terminal_row_y(cfg, "CL2-GND"),
        )
        self.assertLess(
            generator.terminal_row_y(cfg, "CL2-GND"),
            generator.terminal_row_y(cfg, "CL1-GND"),
        )
        self.assertLess(
            generator.terminal_row_y(cfg, "CL1-GND"),
            generator.terminal_row_y(cfg, "CL1"),
        )

    def test_cl1_right_transition_via_labels_match_physical_side(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None

                for turn in range(1, turns + 1):
                    self.assertLess(cl1.points[f"TURN{turn}_RIGHT_UPPER_VIA"][1], 0.0)
                    self.assertGreater(cl1.points[f"TURN{turn}_RIGHT_LOWER_VIA"][1], 0.0)

    def test_cl1_right_transition_uses_generic_lower_rack_topology(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                overrides = {"number_of_secondary_turns": turns}
                if turns >= 4:
                    overrides["target_y_mm"] = 13.0 if turns == 5 else 12.0
                cfg = generator.build_config(overrides)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None

                outer_column = generator.cl1_right_end_column(cfg, 0)
                lower_rack_columns = {
                    cl1.points[f"TURN{turn}_RIGHT_LOWER_VIA"][0]
                    for turn in range(1, turns + 1)
                }
                self.assertEqual(lower_rack_columns, {outer_column})

                for turn_index in range(turns):
                    turn = turn_index + 1
                    reverse_index = turns - 1 - turn_index
                    upper_via = cl1.points[f"TURN{turn}_RIGHT_UPPER_VIA"]
                    lower_via = cl1.points[f"TURN{turn}_RIGHT_LOWER_VIA"]
                    inner_entry_jog = cl1.points[f"TURN{turn}_RIGHT_INNER_ENTRY_JOG"]
                    crossover_jog = cl1.points[f"TURN{turn}_RIGHT_CROSSOVER_JOG"]
                    return_jog = cl1.points[f"TURN{turn}_RIGHT_RETURN_JOG"]
                    reverse_start = cl1.points[f"TURN{turn}_REV_START"]

                    self.assertAlmostEqual(
                        upper_via[0], generator.cl1_right_end_column(cfg, turn_index)
                    )
                    self.assertLess(upper_via[1], 0.0)
                    self.assertGreater(lower_via[1], 0.0)
                    self.assertGreaterEqual(lower_via[0], outer_column)

                    right_end = cl1.points[f"TURN{turn}_RIGHT_END"]
                    self.assertAlmostEqual(
                        right_end[0], generator.cl1_right_end_column(cfg, reverse_index)
                    )
                    self.assertEqual(inner_entry_jog, (right_end[0], lower_via[1]))
                    self.assertEqual(crossover_jog, (upper_via[0], lower_via[1]))
                    self.assertIn((right_end, inner_entry_jog), cl1.inner_segments)
                    if inner_entry_jog != lower_via:
                        self.assertIn((inner_entry_jog, lower_via), cl1.inner_segments)
                    if lower_via != crossover_jog:
                        self.assertIn((lower_via, crossover_jog), cl1.crossover_segments)
                    self.assertIn((crossover_jog, upper_via), cl1.crossover_segments)

                    self.assertEqual(return_jog, reverse_start)
                    self.assertAlmostEqual(reverse_start[0], upper_via[0])
                    self.assertIn((upper_via, return_jog), cl1.target_segments)

                    reverse_mid_end = cl1.points[f"TURN{turn}_REV_MID_END"]
                    reverse_mid_via = cl1.points[f"TURN{turn}_REV_MID_VIA"]
                    reverse_inner_start = cl1.points[f"TURN{turn}_REV_INNER_START"]
                    self.assertIn((reverse_mid_end, reverse_mid_via), cl1.target_segments)
                    self.assertIn((reverse_mid_via, reverse_inner_start), cl1.inner_segments)
                    self.assertTrue(
                        any(segment[0] == reverse_start for segment in cl1.target_segments),
                        f"turn {turn} right return should begin on the target layer",
                    )
                    self.assertTrue(
                        any(segment[0] == reverse_inner_start for segment in cl1.inner_segments),
                        f"turn {turn} left return should begin on Inner.1",
                    )

    def test_cl1_right_transition_prefers_single_rack_when_height_allows(self) -> None:
        cfg = generator.build_config(
            {"number_of_secondary_turns": 5, "target_y_mm": 13.0}
        )
        primary = generator.build_primary_geometry(cfg)
        cl2 = generator.build_cl2_geometry(cfg, primary)
        cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
        assert cl1 is not None

        lower_rack_columns = {
            cl1.points[f"TURN{turn}_RIGHT_LOWER_VIA"][0]
            for turn in range(1, 6)
        }
        self.assertEqual(lower_rack_columns, {generator.cl1_right_end_column(cfg, 0)})

    def test_multiturn_receivers_mirror_generated_points_on_bottom_right_fanout(self) -> None:
        left_cfg = generator.build_config(
            {
                "number_of_secondary_turns": 4,
                "target_y_mm": 12.0,
                "target_side": "bottom",
                "fanout_side": "left",
            }
        )
        right_cfg = generator.build_config(
            {
                "number_of_secondary_turns": 4,
                "target_y_mm": 12.0,
                "target_side": "bottom",
                "fanout_side": "right",
            }
        )
        left_primary = generator.build_primary_geometry(left_cfg)
        right_primary = generator.build_primary_geometry(right_cfg)
        left_cl2 = generator.build_cl2_geometry(left_cfg, left_primary)
        right_cl2 = generator.build_cl2_geometry(right_cfg, right_primary)
        left_cl1 = generator.build_cl1_geometry(left_cfg, left_primary, left_cl2)
        right_cl1 = generator.build_cl1_geometry(right_cfg, right_primary, right_cl2)
        assert left_cl2 is not None and right_cl2 is not None
        assert left_cl1 is not None and right_cl1 is not None

        self.assertEqual(right_cl2.target_layer, "B.Cu")
        self.assertEqual(right_cl2.inner_layer, "In2.Cu")
        self.assertEqual(right_cl1.target_layer, "B.Cu")
        self.assertEqual(right_cl1.inner_layer, "In2.Cu")
        self.assertEqual(right_cl1.crossover_layer, "In1.Cu")
        for name in (
            "TURN1_START",
            "TURN1_LEFT_OUTER",
            "TURN2_LEFT_DETOUR_VIA",
            "TURN3_LEFT_END",
            "TURN2_RIGHT_OUTER",
            "TURN4_RIGHT_DETOUR_VIA",
            "TURN4_RETURN_START",
            "ZP",
        ):
            self.assertAlmostEqual(right_cl2.points[name][0], -left_cl2.points[name][0])
            self.assertAlmostEqual(right_cl2.points[name][1], left_cl2.points[name][1])
        for name in ("TURN1_START", "TURN2_FWD_MID_END", "TURN3_LEFT_UPPER_VIA", "TURN4_RIGHT_UPPER_VIA", "ZN"):
            self.assertAlmostEqual(right_cl1.points[name][0], -left_cl1.points[name][0])
            self.assertAlmostEqual(right_cl1.points[name][1], left_cl1.points[name][1])

    def test_cl2_corrected_u_layer_jump_and_continuity_anchors(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        dimensions = generator.calculate_dimensions(cfg)
        points = generator.build_cl2_point_map(cfg, dimensions)
        target_segments, inner_segments = generator.build_cl2_segments(cfg, dimensions, points)
        half_pitch = generator.trace_pitch(cfg) / 2.0

        self.assertIn((points["T"], points["U"]), inner_segments)
        self.assertIn((points["U"], points["V"]), target_segments)
        self.assertEqual(points["T"], points["V"])
        transition_station_x = (
            points["W"][0]
            - (
                generator.fanout_direction(cfg)
                * cfg["secondary_jump_runup_via_multiplier"]
                * cfg["via_diameter_mm"]
            )
        )
        expected_t = generator.secondary_corrected_rail_point(
            cfg,
            dimensions,
            transition_station_x,
            1.0,
            -half_pitch,
            points["S"],
            points["W"],
        )
        self.assertEqual(points["T"], expected_t)
        self.assertTrue(any(start == points["S"] for start, _ in inner_segments))
        self.assertTrue(any(end == points["T"] for _, end in inner_segments))
        self.assertTrue(any(start == points["V"] for start, _ in target_segments))
        self.assertTrue(any(end == points["W"] for _, end in target_segments))
        self.assertIn((points["ZN"], points["ZO"]), inner_segments)
        self.assertIn((points["ZO"], points["ZP"]), inner_segments)
        inner_curve = generator.secondary_curve_segments(
            cfg,
            dimensions,
            points["S"],
            points["T"],
            1.0,
            -half_pitch,
            points["S"],
            points["W"],
            points["S"][0],
            transition_station_x,
        )
        target_curve = generator.secondary_curve_segments(
            cfg,
            dimensions,
            points["V"],
            points["W"],
            1.0,
            -half_pitch,
            points["S"],
            points["W"],
            transition_station_x,
            points["W"][0],
        )
        self.assertEqual(inner_curve[-1][1], points["T"])
        self.assertEqual(target_curve[0][0], points["V"])

    def test_cl2_paired_vias_use_annular_clearance_spacing(self) -> None:
        cfg = generator.build_config()
        primary = generator.build_primary_geometry(cfg)
        dimensions = generator.calculate_dimensions(cfg)
        points = generator.build_cl2_point_map(cfg, dimensions)
        expected_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
        pitch = generator.trace_pitch(cfg)
        expected_primary_clearance = generator.osc1_via_trace_clearance(cfg)
        inner_primary_y = generator.primary_inner_half_height(cfg, primary.dimensions)

        for first, second in (("E", "Y"), ("H", "ZB"), ("O", "ZI"), ("R", "ZL")):
            self.assertAlmostEqual(
                generator.distance(points[first], points[second]),
                expected_spacing,
            )
        self.assertEqual(points["ZC"][1], points["G"][1])
        self.assertAlmostEqual(points["ZC"][1] - points["ZA"][1], pitch)
        self.assertAlmostEqual(points["ZG"][1] - points["J"][1], pitch)
        self.assertAlmostEqual(points["ZG"][1], -(points["J"][1]))
        self.assertAlmostEqual(
            points["H"][1],
            inner_primary_y - expected_primary_clearance,
        )
        self.assertAlmostEqual(
            points["E"][1],
            -(inner_primary_y - expected_primary_clearance),
        )
        for via_label in ("E", "Y", "H", "ZB", "O", "ZI", "R", "ZL"):
            nearest_primary_trace = min(
                generator.point_to_segment_distance(points[via_label], segment)
                for coil in primary.coils
                for segment in coil.body_segments
            )
            self.assertGreaterEqual(
                nearest_primary_trace + generator.GEOMETRY_TOLERANCE_MM,
                expected_primary_clearance,
            )

    def test_cl2_long_parallel_sinusoidal_rails_preserve_pitch(self) -> None:
        cfg = generator.build_config()
        dimensions = generator.calculate_dimensions(cfg)
        cl2 = generator.build_cl2_geometry(cfg)
        assert cl2 is not None
        points = cl2.points
        half_pitch = generator.trace_pitch(cfg) / 2.0

        parallel_pairs = (
            (
                generator.secondary_curve_segments(
                    cfg, dimensions, points["F"], points["G"], -1.0, half_pitch
                ),
                generator.secondary_curve_segments(
                    cfg, dimensions, points["Z"], points["ZA"], -1.0, -half_pitch
                ),
            ),
            (
                generator.secondary_curve_segments(
                    cfg, dimensions, points["P"], points["Q"], 1.0, half_pitch
                ),
                generator.secondary_curve_segments(
                    cfg, dimensions, points["ZJ"], points["ZK"], 1.0, -half_pitch
                ),
            ),
        )
        for first, second in parallel_pairs:
            self.assertGreaterEqual(
                generator.path_to_path_distance(first, second) + 0.001,
                generator.trace_pitch(cfg),
            )

    def test_cl2_bottom_layers_and_right_fanout_are_mirrored(self) -> None:
        left = generator.build_cl2_geometry(
            generator.build_config({"target_side": "bottom", "fanout_side": "left"})
        )
        right = generator.build_cl2_geometry(
            generator.build_config({"target_side": "bottom", "fanout_side": "right"})
        )
        assert left is not None and right is not None

        self.assertEqual(right.target_layer, "B.Cu")
        self.assertEqual(right.inner_layer, "In2.Cu")
        for name in ("A", "B", "C", "D", "J", "U", "ZN", "ZO", "ZP"):
            self.assertAlmostEqual(right.points[name][0], -left.points[name][0])
            self.assertAlmostEqual(right.points[name][1], left.points[name][1])

    def test_cl2_can_be_disabled_independently(self) -> None:
        cfg = generator.build_config({"generate_cl2": False})

        self.assertIsNone(generator.build_cl2_geometry(cfg))
        self.assertNotIn('(pad "CL2" thru_hole', generator.render_footprint(cfg))

    def test_cl2_rejects_out_of_range_turn_counts_and_coarse_sampling(self) -> None:
        for invalid_turns in (0, 6, 1.5, "3"):
            with self.subTest(invalid_turns=invalid_turns):
                with self.assertRaisesRegex(
                    ValueError,
                    "number_of_secondary_turns must be an integer between 1 and 5",
                ):
                    generator.build_cl2_geometry(
                        generator.build_config({"number_of_secondary_turns": invalid_turns})
                    )
        with self.assertRaisesRegex(ValueError, "integer >= 16"):
            generator.build_cl2_geometry(
                generator.build_config({"secondary_curve_samples_per_cycle": 8})
            )

    def test_default_cl1_span_layers_corrected_vias_and_arcs(self) -> None:
        cl1 = generator.build_cl1_geometry(generator.build_config({"target_x_mm": 21.0, "target_y_mm": 9.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3}))
        assert cl1 is not None

        self.assertEqual(cl1.target_layer, "F.Cu")
        self.assertEqual(cl1.inner_layer, "In1.Cu")
        self.assertEqual(cl1.crossover_layer, "In2.Cu")
        self.assertEqual(cl1.stroke_length_mm, 51.0)
        self.assertIn("D", cl1.via_labels)
        self.assertIn("X", cl1.via_labels)
        self.assertNotIn("Z", cl1.via_labels)
        self.assertNotIn("J", cl1.via_labels)
        self.assertNotIn("ZE", cl1.via_labels)
        self.assertIn("K", cl1.via_labels)
        self.assertIn("L", cl1.via_labels)
        self.assertIn("ZB", cl1.via_labels)
        self.assertIn("ZC", cl1.via_labels)
        self.assertIn((cl1.points["C"], cl1.points["D"]), cl1.crossover_segments)
        self.assertIn((cl1.points["T"], cl1.points["U"]), cl1.crossover_segments)
        self.assertIn((cl1.points["K"], cl1.points["L"]), cl1.crossover_segments)
        self.assertIn((cl1.points["ZB"], cl1.points["ZC"]), cl1.crossover_segments)
        self.assertEqual(len(cl1.target_arcs), 0)
        self.assertEqual(len(cl1.inner_arcs), 1)

    def test_cl1_point_map_is_continuous_across_crossover_and_arc_transitions(self) -> None:
        cl1 = generator.build_cl1_geometry()
        assert cl1 is not None

        self.assertIn((cl1.points["A"], cl1.points["B"]), cl1.target_segments)
        self.assertIn((cl1.points["D"], cl1.points["E"]), cl1.target_segments)
        self.assertIn((cl1.points["G"], cl1.points["H"]), cl1.inner_segments)
        self.assertIn((cl1.points["I"], cl1.points["K"]), cl1.inner_segments)
        self.assertIn((cl1.points["K"], cl1.points["L"]), cl1.crossover_segments)
        self.assertIn((cl1.points["L"], cl1.points["O"]), cl1.target_segments)
        self.assertIn((cl1.points["Z"], cl1.points["ZB"]), cl1.inner_segments)
        self.assertIn((cl1.points["ZB"], cl1.points["ZC"]), cl1.crossover_segments)
        self.assertIn((cl1.points["ZC"], cl1.points["ZF"]), cl1.target_segments)
        self.assertEqual(cl1.inner_arcs[0][0], cl1.points["ZK"])
        self.assertEqual(cl1.inner_arcs[0][2], cl1.points["ZL"])
        self.assertIn((cl1.points["ZL"], cl1.points["ZM"]), cl1.inner_segments)
        self.assertIn((cl1.points["ZM"], cl1.points["ZN"]), cl1.inner_segments)

    def test_cl1_right_end_vertical_turns_align_with_cl2_and_use_via_pitch_columns(self) -> None:
        cfg = generator.build_config({"fanout_side": "left", "target_x_mm": 21.0, "target_y_mm": 9.0, "stroke_range_mm": 51.0, "number_of_primary_turns": 3})
        cl2 = generator.build_cl2_geometry(cfg)
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None and cl2 is not None
        expected_turn_pitch = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]

        self.assertAlmostEqual(cl1.points["K"][0], cl2.points["J"][0])
        self.assertAlmostEqual(cl1.points["L"][0], cl2.points["J"][0])
        self.assertAlmostEqual(cl1.points["ZB"][0], cl1.points["K"][0] - expected_turn_pitch)
        self.assertAlmostEqual(cl1.points["ZC"][0], cl1.points["K"][0] - expected_turn_pitch)
        self.assertAlmostEqual(cl1.points["I"][0], cl1.points["K"][0])
        self.assertAlmostEqual(cl1.points["L"][0], cl1.points["O"][0])
        self.assertAlmostEqual(cl1.points["Z"][0], cl1.points["ZB"][0])
        self.assertAlmostEqual(cl1.points["ZC"][0], cl1.points["ZF"][0])
        self.assertIn((cl1.points["I"], cl1.points["K"]), cl1.inner_segments)
        self.assertIn((cl1.points["L"], cl1.points["O"]), cl1.target_segments)
        self.assertIn((cl1.points["Z"], cl1.points["ZB"]), cl1.inner_segments)
        self.assertIn((cl1.points["ZC"], cl1.points["ZF"]), cl1.target_segments)
        self.assertGreater(cl1.points["K"][1], 0.0)
        self.assertLess(cl1.points["L"][1], 0.0)
        self.assertGreater(cl1.points["ZB"][1], 0.0)
        self.assertLess(cl1.points["ZC"][1], 0.0)

    def test_cl1_zk_zl_arc_is_concentric_with_c_via(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None
        center = cl1.points["C"]
        arc = cl1.inner_arcs[0]
        expected_radius = generator.osc1_via_trace_clearance(cfg)

        self.assertEqual(arc[0], cl1.points["ZK"])
        self.assertEqual(arc[2], cl1.points["ZL"])
        for point in arc:
            self.assertAlmostEqual(generator.distance(point, center), expected_radius)
        self.assertLess(arc[1][0], center[0])
        self.assertLess(arc[1][1], center[1])

    def test_cl1_vertical_turn_vias_clear_cl2_and_stay_inside_osc2_window(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        cl2 = generator.build_cl2_geometry(cfg)
        cl1 = generator.build_cl1_geometry(cfg)
        primary = generator.build_primary_geometry(cfg)
        assert cl1 is not None and cl2 is not None
        via_clearance = generator.osc1_via_trace_clearance(cfg)
        max_turn_half_height = (
            generator.primary_inner_half_height(cfg, primary.dimensions)
            - generator.trace_pitch(cfg)
        )
        cl2_segments = cl2.target_segments + cl2.inner_segments

        for label in ("K", "L", "ZB", "ZC"):
            nearest_trace = min(
                generator.point_to_segment_distance(cl1.points[label], segment)
                for segment in cl2_segments
            )
            self.assertGreaterEqual(
                nearest_trace + generator.GEOMETRY_TOLERANCE_MM,
                via_clearance,
            )
            self.assertLessEqual(abs(cl1.points[label][1]), max_turn_half_height)

    def test_cl1_quadrature_curves_preserve_spacing_across_sampled_runs(self) -> None:
        cfg = generator.build_config()
        dimensions = generator.calculate_dimensions(cfg)
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None
        points = cl1.points
        pitch = generator.trace_pitch(cfg)
        half_pitch = pitch / 2.0
        phase = 3.141592653589793 / 2.0
        half_span = generator.secondary_stroke_length(cfg) / 2.0
        direction = generator.fanout_direction(cfg)
        left_x = direction * half_span
        outer_turn_x, next_turn_x = generator.cl1_right_end_columns(cfg)
        outer_turn_x *= -direction
        next_turn_x *= -direction
        via_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
        midpoint_left_x = direction * (via_spacing / 2.0)
        midpoint_right_x = -direction * (via_spacing / 2.0)
        transition_x = left_x - direction * (
            generator.secondary_stroke_length(cfg)
            * cfg["cl1_transition_column_fraction"]
        )
        station_x_map = {
            "E": left_x, "F": midpoint_left_x,
            "H": midpoint_left_x, "I": outer_turn_x,
            "O": outer_turn_x, "P": midpoint_right_x,
            "R": midpoint_right_x, "S": transition_x,
            "V": transition_x, "W": midpoint_right_x,
            "Y": midpoint_right_x, "Z": next_turn_x,
            "ZF": next_turn_x, "ZG": midpoint_left_x,
            "ZI": midpoint_left_x, "ZJ": left_x,
        }
        curve_pairs = (
            (("E", "F", 1.0, half_pitch), ("V", "W", 1.0, -half_pitch)),
            (("H", "I", 1.0, -half_pitch), ("Y", "Z", 1.0, half_pitch)),
            (("O", "P", -1.0, -half_pitch), ("ZF", "ZG", -1.0, half_pitch)),
            (("R", "S", -1.0, half_pitch), ("ZI", "ZJ", -1.0, -half_pitch)),
        )
        for first, second in curve_pairs:
            first_path = generator.secondary_curve_segments(
                cfg,
                dimensions,
                points[first[0]],
                points[first[1]],
                first[2],
                first[3],
                station_start_x=station_x_map[first[0]],
                station_end_x=station_x_map[first[1]],
                phase_offset_radians=phase,
                mirror_phase_sign=False,
            )
            second_path = generator.secondary_curve_segments(
                cfg,
                dimensions,
                points[second[0]],
                points[second[1]],
                second[2],
                second[3],
                station_start_x=station_x_map[second[0]],
                station_end_x=station_x_map[second[1]],
                phase_offset_radians=phase,
                mirror_phase_sign=False,
            )
            self.assertGreaterEqual(
                generator.path_to_path_distance(first_path, second_path) + 0.003,
                generator.trace_pitch(cfg),
            )

    def test_cl1_bottom_layers_and_right_fanout_are_mirrored(self) -> None:
        left = generator.build_cl1_geometry(
            generator.build_config({"target_side": "bottom", "fanout_side": "left"})
        )
        right = generator.build_cl1_geometry(
            generator.build_config({"target_side": "bottom", "fanout_side": "right"})
        )
        assert left is not None and right is not None

        self.assertEqual(right.target_layer, "B.Cu")
        self.assertEqual(right.inner_layer, "In2.Cu")
        self.assertEqual(right.crossover_layer, "In1.Cu")
        for name in ("A", "B", "C", "D", "I", "K", "L", "Z", "ZB", "ZC", "ZF", "T", "U", "ZM", "ZN"):
            self.assertAlmostEqual(right.points[name][0], -left.points[name][0])
            self.assertAlmostEqual(right.points[name][1], left.points[name][1])

    def test_cl1_can_be_disabled_and_rejects_tight_primary_endpoint(self) -> None:
        cfg = generator.build_config({"generate_cl1": False})
        self.assertIsNone(generator.build_cl1_geometry(cfg))
        self.assertNotIn('(pad "CL1" thru_hole', generator.render_footprint(cfg))

        tight_cfg = generator.build_config({"primary_end_extension_mm": 0.1})
        with self.assertRaisesRegex(ValueError, "CL1 endpoint"):
            generator.build_cl1_geometry(tight_cfg)


if __name__ == "__main__":
    unittest.main()
