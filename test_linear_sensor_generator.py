import math
import unittest

import linear_sensor_generator as generator


class LinearSensorGeneratorTests(unittest.TestCase):
    def test_reference_dimensions_and_primary_layers(self) -> None:
        cfg = generator.build_config({"target_y_mm": 7.0})
        geometry = generator.build_primary_geometry(cfg)
        osc1, osc2 = geometry.coils

        self.assertEqual(geometry.dimensions.secondary_period_mm, 42.0)
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
        cfg = generator.build_config({"fanout_side": "left"})
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
        self.assertEqual(footprint.count("(fp_arc "), 5)
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
        cfg = generator.build_config({"fanout_side": "left"})
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
        cfg = generator.build_config({"target_y_mm": 7.0, "fanout_side": "left"})
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
        cfg = generator.build_config({"target_y_mm": 10.0})

        with self.assertRaisesRegex(ValueError, "CL1 parallel sinusoidal traces"):
            generator.render_footprint(cfg)

    def test_cl2_corrected_u_layer_jump_and_continuity_anchors(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        dimensions = generator.calculate_dimensions(cfg)
        cl2 = generator.build_cl2_geometry(cfg)
        assert cl2 is not None
        points = cl2.points
        half_pitch = generator.trace_pitch(cfg) / 2.0

        self.assertIn((points["T"], points["U"]), cl2.inner_segments)
        self.assertIn((points["U"], points["V"]), cl2.target_segments)
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
        self.assertTrue(any(start == points["S"] for start, _ in cl2.inner_segments))
        self.assertTrue(any(end == points["T"] for _, end in cl2.inner_segments))
        self.assertTrue(any(start == points["V"] for start, _ in cl2.target_segments))
        self.assertTrue(any(end == points["W"] for _, end in cl2.target_segments))
        self.assertIn((points["ZN"], points["ZO"]), cl2.inner_segments)
        self.assertIn((points["ZO"], points["ZP"]), cl2.inner_segments)
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
        cl2 = generator.build_cl2_geometry(cfg)
        assert cl2 is not None
        expected_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
        pitch = generator.trace_pitch(cfg)
        expected_primary_clearance = generator.osc1_via_trace_clearance(cfg)
        inner_primary_y = generator.primary_inner_half_height(cfg, primary.dimensions)

        for first, second in (("E", "Y"), ("H", "ZB"), ("O", "ZI"), ("R", "ZL")):
            self.assertAlmostEqual(
                generator.distance(cl2.points[first], cl2.points[second]),
                expected_spacing,
            )
        self.assertEqual(cl2.points["ZC"][1], cl2.points["G"][1])
        self.assertAlmostEqual(cl2.points["ZC"][1] - cl2.points["ZA"][1], pitch)
        self.assertAlmostEqual(cl2.points["ZG"][1] - cl2.points["J"][1], pitch)
        self.assertAlmostEqual(cl2.points["ZG"][1], -(cl2.points["J"][1]))
        self.assertAlmostEqual(
            cl2.points["H"][1],
            inner_primary_y - expected_primary_clearance,
        )
        self.assertAlmostEqual(
            cl2.points["E"][1],
            -(inner_primary_y - expected_primary_clearance),
        )
        for via_label in ("E", "Y", "H", "ZB", "O", "ZI", "R", "ZL"):
            nearest_primary_trace = min(
                generator.point_to_segment_distance(cl2.points[via_label], segment)
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

    def test_cl2_rejects_unmapped_turn_counts_and_coarse_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two secondary turns"):
            generator.build_cl2_geometry(generator.build_config({"number_of_secondary_turns": 3}))
        with self.assertRaisesRegex(ValueError, "integer >= 16"):
            generator.build_cl2_geometry(
                generator.build_config({"secondary_curve_samples_per_cycle": 8})
            )

    def test_default_cl1_span_layers_corrected_vias_and_arcs(self) -> None:
        cl1 = generator.build_cl1_geometry()
        assert cl1 is not None

        self.assertEqual(cl1.target_layer, "F.Cu")
        self.assertEqual(cl1.inner_layer, "In1.Cu")
        self.assertEqual(cl1.crossover_layer, "In2.Cu")
        self.assertEqual(cl1.stroke_length_mm, 71.0)
        self.assertIn("D", cl1.via_labels)
        self.assertIn("X", cl1.via_labels)
        self.assertNotIn("Z", cl1.via_labels)
        self.assertIn((cl1.points["C"], cl1.points["D"]), cl1.crossover_segments)
        self.assertIn((cl1.points["T"], cl1.points["U"]), cl1.crossover_segments)
        self.assertEqual(len(cl1.target_arcs), 2)
        self.assertEqual(len(cl1.inner_arcs), 3)

    def test_cl1_point_map_is_continuous_across_crossover_and_arc_transitions(self) -> None:
        cl1 = generator.build_cl1_geometry()
        assert cl1 is not None

        self.assertIn((cl1.points["A"], cl1.points["B"]), cl1.target_segments)
        self.assertIn((cl1.points["D"], cl1.points["E"]), cl1.target_segments)
        self.assertIn((cl1.points["G"], cl1.points["H"]), cl1.inner_segments)
        self.assertIn((cl1.points["I"], cl1.points["J"]), cl1.inner_segments)
        self.assertEqual(cl1.target_arcs[0][0], cl1.points["K"])
        self.assertEqual(cl1.target_arcs[0][2], cl1.points["L"])
        self.assertEqual(cl1.inner_arcs[0][0], cl1.points["ZA"])
        self.assertEqual(cl1.inner_arcs[0][2], cl1.points["ZB"])
        self.assertIn((cl1.points["ZL"], cl1.points["ZM"]), cl1.inner_segments)
        self.assertIn((cl1.points["ZM"], cl1.points["ZN"]), cl1.inner_segments)

    def test_cl1_mn_and_zazb_arcs_preserve_columns_around_cl1_vias(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None
        expected_radius = generator.osc1_via_trace_clearance(cfg)
        expected_inner_x = 35.5 - generator.trace_pitch(cfg)

        for arc, via_label in (
            (cl1.target_arcs[1], "ZE"),
            (cl1.inner_arcs[0], "J"),
        ):
            center = cl1.points[via_label]
            for point in arc:
                self.assertAlmostEqual(generator.distance(point, center), expected_radius)
            self.assertGreater(arc[1][0], center[0])
        self.assertAlmostEqual(cl1.points["M"][0], expected_inner_x)
        self.assertAlmostEqual(cl1.points["N"][0], 35.5)
        self.assertAlmostEqual(cl1.points["ZA"][0], 35.5)
        self.assertAlmostEqual(cl1.points["ZB"][0], expected_inner_x)

    def test_cl1_zk_zl_arc_is_concentric_with_c_via(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None
        center = cl1.points["C"]
        arc = cl1.inner_arcs[2]
        expected_radius = generator.osc1_via_trace_clearance(cfg)

        self.assertEqual(arc[0], cl1.points["ZK"])
        self.assertEqual(arc[2], cl1.points["ZL"])
        for point in arc:
            self.assertAlmostEqual(generator.distance(point, center), expected_radius)
        self.assertLess(arc[1][0], center[0])
        self.assertLess(arc[1][1], center[1])

    def test_cl1_kl_and_zczd_arcs_are_centered_on_column_with_cl2_clearance(self) -> None:
        cfg = generator.build_config({"fanout_side": "left"})
        cl2 = generator.build_cl2_geometry(cfg)
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None and cl2 is not None
        pitch = generator.trace_pitch(cfg)

        for arc in (cl1.target_arcs[0], cl1.inner_arcs[1]):
            center = (arc[0][0], (arc[0][1] + arc[2][1]) / 2.0)
            radius = generator.distance(center, arc[0])
            self.assertAlmostEqual(center[0], cl1.points["K"][0])
            self.assertAlmostEqual(center[1], 0.0)
            self.assertAlmostEqual(arc[1][0], center[0] + radius)
            self.assertAlmostEqual(arc[1][1], center[1])
            for label in ("J", "ZG"):
                self.assertGreaterEqual(
                    radius - generator.distance(center, cl2.points[label])
                    + generator.GEOMETRY_TOLERANCE_MM,
                    pitch,
                )

    def test_cl1_quadrature_curves_preserve_spacing_across_sampled_runs(self) -> None:
        cfg = generator.build_config()
        dimensions = generator.calculate_dimensions(cfg)
        cl1 = generator.build_cl1_geometry(cfg)
        assert cl1 is not None
        points = cl1.points
        half_pitch = generator.trace_pitch(cfg) / 2.0
        phase = 3.141592653589793 / 2.0
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
                phase_offset_radians=phase,
                mirror_phase_sign=False,
            )
            self.assertGreaterEqual(
                generator.path_to_path_distance(first_path, second_path) + 0.002,
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
        for name in ("A", "B", "C", "D", "J", "T", "U", "ZM", "ZN"):
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
