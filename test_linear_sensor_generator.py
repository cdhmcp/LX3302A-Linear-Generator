import unittest

import linear_sensor_generator as generator


class LinearSensorGeneratorTests(unittest.TestCase):
    def test_default_dimensions_and_top_target_layers(self) -> None:
        geometry = generator.build_primary_geometry()

        self.assertEqual(geometry.dimensions.secondary_period_mm, 42.0)
        self.assertEqual(geometry.dimensions.secondary_length_mm, 91.0)
        self.assertEqual(geometry.dimensions.secondary_width_mm, 5.5)
        self.assertEqual(geometry.dimensions.primary_length_mm, 97.0)
        self.assertEqual(geometry.dimensions.primary_width_mm, 11.5)
        self.assertEqual(tuple(coil.layer for coil in geometry.coils), ("B.Cu", "In2.Cu"))
        self.assertEqual(set(geometry.coils[0].body_segments), set(geometry.coils[1].body_segments))
        self.assertNotEqual(geometry.coils[0].fanout_segments[0][0], geometry.coils[1].fanout_segments[0][0])

    def test_default_footprint_has_two_coils_and_named_vias(self) -> None:
        footprint = generator.render_footprint()

        self.assertIn('(pad "VIN" thru_hole', footprint)
        self.assertIn('(pad "OSC1" thru_hole', footprint)
        self.assertIn('(pad "OSC2" thru_hole', footprint)
        self.assertIn('(layer "B.Cu")', footprint)
        self.assertIn('(layer "In2.Cu")', footprint)
        self.assertEqual(footprint.count('(layer "B.Cu")'), 12)
        self.assertEqual(footprint.count('(layer "In2.Cu")'), 12)

    def test_bottom_target_and_right_fanout_flip_layers_and_pads(self) -> None:
        cfg = generator.build_config({"target_side": "bottom", "fanout_side": "right"})
        geometry = generator.build_primary_geometry(cfg)

        self.assertEqual(tuple(coil.layer for coil in geometry.coils), ("F.Cu", "In1.Cu"))
        self.assertGreater(geometry.pads["VIN"][0], 0.0)
        self.assertEqual(geometry.dimensions.primary_length_mm, 97.0)

    def test_alternate_dimensions_and_turn_count_generate(self) -> None:
        cfg = generator.build_config(
            {
                "target_x_mm": 15.0,
                "target_y_mm": 8.0,
                "number_of_primary_turns": 3,
            }
        )
        geometry = generator.build_primary_geometry(cfg)

        self.assertEqual(geometry.dimensions.secondary_period_mm, 30.0)
        self.assertEqual(geometry.dimensions.secondary_width_mm, 6.5)
        self.assertEqual(len(geometry.coils[0].body_segments), 13)

    def test_invalid_secondary_width_is_rejected(self) -> None:
        cfg = generator.build_config({"secondary_y_reduction_mm": 7.0})

        with self.assertRaisesRegex(ValueError, "positive secondary width"):
            generator.build_primary_geometry(cfg)

    def test_excessive_turn_count_is_rejected(self) -> None:
        cfg = generator.build_config({"number_of_primary_turns": 30})

        with self.assertRaisesRegex(ValueError, "Primary width is insufficient"):
            generator.build_primary_geometry(cfg)

    def test_invalid_dimension_is_rejected(self) -> None:
        cfg = generator.build_config({"target_x_mm": 0.0})

        with self.assertRaisesRegex(ValueError, "target_x_mm"):
            generator.build_primary_geometry(cfg)


if __name__ == "__main__":
    unittest.main()
