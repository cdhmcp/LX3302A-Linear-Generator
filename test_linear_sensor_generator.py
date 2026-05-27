import unittest

import linear_sensor_generator as generator


class LinearSensorGeneratorTests(unittest.TestCase):
    def test_default_dimensions_and_primary_layers(self) -> None:
        geometry = generator.build_primary_geometry()
        osc1, osc2 = geometry.coils

        self.assertEqual(geometry.dimensions.secondary_period_mm, 42.0)
        self.assertEqual(geometry.dimensions.secondary_length_mm, 91.0)
        self.assertEqual(geometry.dimensions.secondary_width_mm, 5.5)
        self.assertEqual(geometry.dimensions.primary_length_mm, 97.0)
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

    def test_default_footprint_emits_osc1_osc2_and_two_vin_vias(self) -> None:
        footprint = generator.render_footprint()

        self.assertIn('(pad "OSC1" thru_hole', footprint)
        self.assertEqual(footprint.count('(pad "VIN" thru_hole'), 2)
        self.assertIn('(pad "OSC2" thru_hole', footprint)
        self.assertIn('(layer "B.Cu")', footprint)
        self.assertIn('(layer "F.Cu")', footprint)
        self.assertIn('(layer "In2.Cu")', footprint)

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


if __name__ == "__main__":
    unittest.main()
