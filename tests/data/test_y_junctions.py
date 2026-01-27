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
        noise_magnitude_range=(8, 25),
        ellipse_ratio_range=(1.1, 1.5),
        use_gpu=False,
        seed=42,
    )
    (segmentation,
    distance_field_raw,
    distance_field,
    distance_field_squared,
    tubular_skeleton,
    skeleton,
    regression_target) = generate_y_junction(*params)

    assert all(a.shape == skeleton.shape for a in (
        distance_field,
        segmentation,
        distance_field_squared,
        tubular_skeleton,
        distance_field_raw,
        regression_target,
    ))
    assert skeleton.dtype ==  "uint8"
    assert distance_field.dtype == "float64"
