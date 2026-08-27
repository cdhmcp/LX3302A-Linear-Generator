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

        self.assertEqual(coil.body_segments[0], (points["A"], points["B"]))
        self.assertEqual(points["A"][1], points["B"][1])
        self.assertEqual(points["A"][1], 0.0)
        self.assertIn((points["B"], points["C"]), coil.body_segments)
        self.assertIn((points["H"], points["I"]), coil.body_segments)
        self.assertIn((points["N"], points["O"]), coil.body_segments)
        self.assertEqual(coil.body_segments[-1], (points["T"], points["U"]))
        self.assertEqual(
            coil.escape_segments,
            ((points["U"], points["VIN_JOG"]), (points["VIN_JOG"], points["V"])),
        )
        self.assertEqual(points["U"][1], points["VIN_JOG"][1])
        self.assertAlmostEqual(
            abs(points["V"][0] - points["VIN_JOG"][0]),
            abs(points["V"][1] - points["VIN_JOG"][1]),
        )
        self.assertAlmostEqual(abs(points["C"][0] - points["B"][0]), pitch / 2.0)
        self.assertAlmostEqual(abs(points["C"][1] - points["B"][1]), pitch / 2.0)
        self.assertAlmostEqual(abs(points["I"][0] - points["H"][0]), pitch)
        self.assertAlmostEqual(abs(points["I"][1] - points["H"][1]), pitch)
        self.assertAlmostEqual(points["J"][0] - points["D"][0], pitch)
        self.assertAlmostEqual(points["P"][0] - points["J"][0], pitch)
        self.assertAlmostEqual(points["N"][1] - points["H"][1], -transition_shift)
        self.assertAlmostEqual(points["C"][1] - points["H"][1], junction_separation)
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
        self.assertEqual(points["A"][1], 0.0)
        self.assertEqual(points["U"][1], -1.2)
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
            ("CL1-GND", cl1.points["ZN"]),
            ("CL2-GND", cl2.points["ZP"]),
            ("VIN", primary.pads["VIN_V"]),
            ("CL1", cl1.points["A"]),
            ("OSC1", primary.pads["OSC1_A"]),
            ("OSC2", primary.pads["OSC2_A"]),
            ("CL2", cl2.points["A"]),
        )
        expected_x = generator.terminal_column_x(cfg, primary.dimensions)
        expected_spacing = generator.terminal_pad_pitch(cfg)

        for name, point in terminals:
            self.assertAlmostEqual(point[0], expected_x)
            self.assertAlmostEqual(point[1], generator.terminal_row_y(cfg, name))
        for (_, first), (_, second) in zip(terminals, terminals[1:]):
            self.assertAlmostEqual(generator.distance(first, second), expected_spacing)
        self.assertEqual(cl1.points["A"][1], cl1.points["B"][1])
        self.assertEqual(cl1.points["B"][1], cl1.points["C"][1])

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
        self.assertAlmostEqual(
            abs(cl1.points["ZN"][0] - cl1.points["ZM"][0]),
            abs(cl1.points["ZN"][1] - cl1.points["ZM"][1]),
        )

        self.assertEqual(cl2.points["A"][0], terminal_x)
        self.assertLess(cl2.points["B"][0], cl2.points["A"][0])
        self.assertGreater(cl2.points["B"][0], sensor_edge_x)
        self.assertAlmostEqual(
            abs(cl2.points["A"][0] - cl2.points["B"][0]),
            abs(cl2.points["A"][1] - cl2.points["B"][1]),
        )
        self.assertEqual(cl2.points["ZP"][0], terminal_x)
        self.assertLess(cl2.points["ZO"][0], cl2.points["ZP"][0])
        self.assertGreater(cl2.points["ZO"][0], sensor_edge_x)
        self.assertAlmostEqual(
            abs(cl2.points["ZP"][0] - cl2.points["ZO"][0]),
            abs(cl2.points["ZP"][1] - cl2.points["ZO"][1]),
        )

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

    def test_exit_offset_moves_internal_via_without_moving_terminal_row(self) -> None:
        cfg = generator.build_config({"osc1_vin_exit_offset_mm": 0.55})
        points = generator.build_primary_geometry(cfg).coils[0].points

        self.assertEqual(points["U"][1], -0.55)
        self.assertEqual(points["VIN_JOG"][1], points["U"][1])
        self.assertAlmostEqual(
            abs(points["V"][0] - points["VIN_JOG"][0]),
            abs(points["V"][1] - points["VIN_JOG"][1]),
        )
        self.assertEqual(points["V"][1], generator.terminal_row_y(cfg, "VIN"))
        self.assertEqual(points["V"][0], points["A"][0])

    def test_exit_offset_below_horizontal_trace_clearance_is_rejected(self) -> None:
        cfg = generator.build_config({"osc1_vin_exit_offset_mm": 0.1})

        with self.assertRaisesRegex(ValueError, "too small for OSC1/VIN"):
            generator.build_primary_geometry(cfg)

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
        self.assertAlmostEqual(
            abs(points["B"][0] - points["A_JOG"][0]),
            abs(points["B"][1] - points["A_JOG"][1]),
        )
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

    def test_multiturn_receivers_build_in_strict_mode_for_one_to_five_turns(self) -> None:
        for turns in range(1, 6):
            with self.subTest(turns=turns):
                cfg = generator.build_config({"number_of_secondary_turns": turns})
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)

                self.assertIsNotNone(cl2)
                self.assertIsNotNone(cl1)

    def test_two_turn_receivers_use_generalized_builders_but_keep_legacy_labels(self) -> None:
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
        self.assertEqual(cl2.via_labels, generator.CL2_TWO_TURN_LEGACY_VIA_LABELS)
        self.assertEqual(cl1.via_labels, generator.CL1_TWO_TURN_LEGACY_VIA_LABELS)

    def test_multiturn_cl2_columns_center_on_quarter_span_and_stay_inside_outer_envelope(self) -> None:
        for turns in (1, 3, 4, 5):
            with self.subTest(turns=turns):
                cfg = generator.build_config({"number_of_secondary_turns": turns})
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

    def test_multiturn_cl1_columns_center_on_midpoint_and_step_inward(self) -> None:
        for turns in (1, 3, 4, 5):
            with self.subTest(turns=turns):
                cfg = generator.build_config({"number_of_secondary_turns": turns})
                dimensions = generator.calculate_dimensions(cfg)
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None
                via_spacing = generator.secondary_via_spacing(cfg)
                half_span = generator.secondary_stroke_length(cfg) / 2.0
                centered_midpoints = [
                    (index - ((turns - 1) / 2.0)) * via_spacing
                    for index in range(turns)
                ]
                left_midpoints = [value for value in centered_midpoints if value < 0.0]
                right_midpoints = [value for value in centered_midpoints if value > 0.0]
                expected_midpoints: list[float] = []
                for left, right in zip(left_midpoints, reversed(right_midpoints)):
                    expected_midpoints.extend((left, right))
                if turns % 2 == 1:
                    expected_midpoints.append(0.0)
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
                if turns > 1:
                    transition_base = (
                        -half_span
                        + (
                            generator.secondary_stroke_length(cfg)
                            * cfg["cl1_transition_column_fraction"]
                        )
                    )
                    actual_transitions = [
                        cl1.points[f"TURN{index + 1}_LEFT_TRANSITION_UPPER_VIA"][0]
                        for index in range(turns - 1)
                    ]
                    expected_transitions = [
                        transition_base + (index * via_spacing)
                        for index in range(turns - 1)
                    ]
                    for actual, expected in zip(actual_transitions, expected_transitions):
                        self.assertAlmostEqual(actual, expected)
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

    def test_multiturn_cl1_left_transition_handoff_stays_off_inner_next_start(self) -> None:
        for turns in (3, 5):
            with self.subTest(turns=turns):
                cfg = generator.build_config({"number_of_secondary_turns": turns})
                primary = generator.build_primary_geometry(cfg)
                cl2 = generator.build_cl2_geometry(cfg, primary)
                cl1 = generator.build_cl1_geometry(cfg, primary, cl2)
                assert cl1 is not None

                inner_points = {
                    point
                    for segment in cl1.inner_segments
                    for point in segment
                }
                for turn_index in range(1, turns):
                    next_start = cl1.points[f"TURN{turn_index + 1}_START"]
                    lower_via = cl1.points[f"TURN{turn_index}_LEFT_TRANSITION_LOWER_VIA"]
                    self.assertIn((lower_via, next_start), cl1.target_segments)
                    self.assertNotIn(next_start, inner_points)

    def test_multiturn_receivers_mirror_generated_points_on_bottom_right_fanout(self) -> None:
        left_cfg = generator.build_config(
            {"number_of_secondary_turns": 4, "target_side": "bottom", "fanout_side": "left"}
        )
        right_cfg = generator.build_config(
            {"number_of_secondary_turns": 4, "target_side": "bottom", "fanout_side": "right"}
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
            "TURN2_RIGHT_OUTER",
            "TURN4_RIGHT_DETOUR_VIA",
            "TURN4_RETURN_START",
            "ZP",
        ):
            self.assertAlmostEqual(right_cl2.points[name][0], -left_cl2.points[name][0])
            self.assertAlmostEqual(right_cl2.points[name][1], left_cl2.points[name][1])
        for name in ("TURN1_START", "TURN2_FWD_MID_END", "TURN3_LEFT_TRANSITION_UPPER_VIA", "TURN4_RIGHT_UPPER_VIA", "ZN"):
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
