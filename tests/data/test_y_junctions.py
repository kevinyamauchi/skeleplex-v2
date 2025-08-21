from skeleplex.data.y_junctions import (
    generate_y_junction,
    random_parameters_y_junctions,
)


def test_generate_y_junction():
    params = random_parameters_y_junctions(
        length_parent_range=(30, 35),
        length_d1_range=(10, 12),
        length_d2_range=(10, 12),
        radius_parent_range=(10, 15),
        radius_d1_range=(10, 15),
        radius_d2_range=(10, 15),
        d1_angle_range=(-45, 45),
        d2_angle_range=(-45, 45),
        wiggle_factor_range=(0.01, 0.03),
        noise_magnitude_range=(8, 25),
        ellipse_ratio_range=(1.1, 1.5),
        use_gpu=False,
        seed=42,
    )
    skeleton, distance_field = generate_y_junction(*params)

    assert skeleton.shape == distance_field.shape
    assert skeleton.dtype == "int"
    assert distance_field.dtype == "float32"
