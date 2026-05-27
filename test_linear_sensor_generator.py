import unittest

import linear_sensor_generator as generator


class LinearSensorGeneratorTests(unittest.TestCase):
    def test_default_dimensions_and_primary_layers(self) -> None:
        geometry = generator.build_primary_geometry()
        osc1, osc2 = geometry.coils

        self.assertEqual(geometry.dimensions.secondary_period_mm, 42.0)
        self.assertEqual(geometry.dimensions.secondary_length_mm, 72.0)
        self.assertEqual(geometry.dimensions.secondary_width_mm, 5.5)
        self.assertEqual(geometry.dimensions.primary_length_mm, 78.0)
        self.assertEqual(geometry.dimensions.primary_width_mm, 11.5)
        self.assertEqual(osc1.name, "OSC1")
        self.assertEqual(osc1.layer, "B.Cu")
        self.assertEqual(osc1.escape_layer, "F.Cu")
        self.assertEqual(osc2.name, "OSC2")
        self.assertEqual(osc2.layer, "In2.Cu")
        self.assertEqual(osc2.escape_segments, ())

    def test_default_osc1_uses_annotated_three_turn_path(self) -> None:
        coil = generator.build_primary_geometry().coils[0]
        points = coil.points
        pitch = generator.trace_pitch(generator.build_config())
        transition_shift = generator.parallel_45_center_shift(generator.build_config())
        junction_separation = generator.parallel_45_junction_separation(generator.build_config())

        self.assertEqual(coil.body_segments[0], (points["A"], points["B"]))
        self.assertEqual(points["A"][1], points["B"][1])
        self.assertEqual(points["A"][1], 0.0)
        self.assertIn((points["B"], points["C"]), coil.body_segments)
        self.assertIn((points["H"], points["I"]), coil.body_segments)
        self.assertIn((points["N"], points["O"]), coil.body_segments)
        self.assertEqual(coil.body_segments[-1], (points["T"], points["U"]))
        self.assertEqual(coil.escape_segments, ((points["U"], points["V"]),))
        self.assertEqual(points["U"][1], points["V"][1])
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
        cfg = generator.build_config()
        points = generator.build_primary_geometry(cfg).coils[0].points
        dimensions = generator.calculate_dimensions(cfg)
        expected_terminal_x = -(dimensions.primary_length_mm / 2.0) - cfg["terminal_escape_length_mm"]

        self.assertEqual(points["A"][0], expected_terminal_x)
        self.assertEqual(points["V"][0], expected_terminal_x)
        self.assertEqual(points["A"][1], 0.0)
        self.assertEqual(points["U"][1], -1.2)
        self.assertEqual(points["V"][1], -1.2)

    def test_default_footprint_emits_oscillators_cl2_and_two_vin_vias(self) -> None:
        footprint = generator.render_footprint()

        self.assertIn('(pad "OSC1" thru_hole', footprint)
        self.assertEqual(footprint.count('(pad "VIN" thru_hole'), 2)
        self.assertIn('(pad "OSC2" thru_hole', footprint)
        self.assertIn('(layer "B.Cu")', footprint)
        self.assertIn('(layer "F.Cu")', footprint)
        self.assertIn('(layer "In2.Cu")', footprint)
        self.assertIn('(pad "CL2" thru_hole', footprint)
        self.assertIn('(pad "CL2-GND" thru_hole', footprint)
        self.assertIn('(layer "In1.Cu")', footprint)

    def test_bottom_target_mirrors_primary_and_escape_layers(self) -> None:
        cfg = generator.build_config({"target_side": "bottom"})
        osc1, osc2 = generator.build_primary_geometry(cfg).coils

        self.assertEqual(osc1.layer, "F.Cu")
        self.assertEqual(osc1.escape_layer, "B.Cu")
        self.assertEqual(osc2.layer, "In1.Cu")

    def test_right_fanout_mirrors_point_map_horizontally(self) -> None:
        left_geometry = generator.build_primary_geometry()
        cfg = generator.build_config({"fanout_side": "right"})
        right_geometry = generator.build_primary_geometry(cfg)

        for name in ("A", "B", "C", "D", "E", "F", "U", "V"):
            self.assertAlmostEqual(right_geometry.coils[0].points[name][0], -left_geometry.coils[0].points[name][0])
            self.assertAlmostEqual(right_geometry.coils[0].points[name][1], left_geometry.coils[0].points[name][1])
        for name in ("A", "B", "C", "D", "E", "F", "X"):
            self.assertAlmostEqual(right_geometry.coils[1].points[name][0], -left_geometry.coils[1].points[name][0])
            self.assertAlmostEqual(right_geometry.coils[1].points[name][1], left_geometry.coils[1].points[name][1])

    def test_configurable_fourth_turn_gets_generated_labels(self) -> None:
        cfg = generator.build_config({"number_of_primary_turns": 4})
        osc1, osc2 = generator.build_primary_geometry(cfg).coils

        self.assertIn("TURN4_START", osc1.points)
        self.assertIn("OSC2_TURN4_START", osc2.points)
        self.assertEqual(osc1.body_segments[-1][1], osc1.points["U"])
        self.assertEqual(osc2.body_segments[-1][1], osc2.points["X"])

    def test_u_via_clearance_is_derived_from_via_and_trace_properties(self) -> None:
        cfg = generator.build_config()
        points = generator.build_primary_geometry(cfg).coils[0].points
        expected_clearance = generator.osc1_via_trace_clearance(cfg)

        self.assertAlmostEqual(points["U"][0] - points["T"][0], expected_clearance)
        self.assertAlmostEqual(abs(points["U"][1] - points["T"][1]), expected_clearance)

    def test_tiny_exit_offset_fans_out_via_without_moving_lane_y(self) -> None:
        cfg = generator.build_config({"osc1_vin_exit_offset_mm": 0.55})
        points = generator.build_primary_geometry(cfg).coils[0].points

        self.assertEqual(points["V"][1], -0.55)
        self.assertLess(points["V"][0], points["A"][0])

    def test_exit_offset_below_horizontal_trace_clearance_is_rejected(self) -> None:
        cfg = generator.build_config({"osc1_vin_exit_offset_mm": 0.1})

        with self.assertRaisesRegex(ValueError, "too small for horizontal"):
            generator.build_primary_geometry(cfg)

    def test_invalid_secondary_width_is_rejected(self) -> None:
        cfg = generator.build_config({"secondary_y_reduction_mm": 7.0})

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

    def test_default_cl2_span_layers_and_outer_extrema(self) -> None:
        cl2 = generator.build_cl2_geometry()
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

    def test_cl2_corrected_u_layer_jump_and_continuity_anchors(self) -> None:
        cfg = generator.build_config()
        dimensions = generator.calculate_dimensions(cfg)
        cl2 = generator.build_cl2_geometry()
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
        cl2 = generator.build_cl2_geometry(cfg)
        assert cl2 is not None
        expected_spacing = cfg["via_diameter_mm"] + cfg["trace_spacing_mm"]
        pitch = generator.trace_pitch(cfg)

        for first, second in (("E", "Y"), ("H", "ZB"), ("O", "ZI"), ("R", "ZL")):
            self.assertAlmostEqual(
                generator.distance(cl2.points[first], cl2.points[second]),
                expected_spacing,
            )
        self.assertEqual(cl2.points["ZC"][1], cl2.points["G"][1])
        self.assertAlmostEqual(cl2.points["ZC"][1] - cl2.points["ZA"][1], pitch)
        self.assertAlmostEqual(cl2.points["ZG"][1] - cl2.points["J"][1], pitch)
        self.assertAlmostEqual(cl2.points["ZG"][1], -(cl2.points["J"][1]))

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
        left = generator.build_cl2_geometry()
        right = generator.build_cl2_geometry(
            generator.build_config({"target_side": "bottom", "fanout_side": "right"})
        )
        assert left is not None and right is not None

        self.assertEqual(right.target_layer, "B.Cu")
        self.assertEqual(right.inner_layer, "In2.Cu")
        for name in ("A", "C", "D", "J", "U", "ZP"):
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


if __name__ == "__main__":
    unittest.main()
