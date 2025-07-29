from skeleplex.data.fractal_trees import (
    generate_random_parameters_for_fractal_tree,
    generate_synthetic_fractal_tree,
)


def test_generate_fractal_tree():
    params = generate_random_parameters_for_fractal_tree(
        num_nodes_range=(5, 6),
        edge_length_factor=(5, 6),
        branch_angle_range=(30, 90),
        wiggle_factor_range=(0.01, 0.03),
        noise_magnitude_range=(8, 25),
        ellipse_ratio_range=(1.1, 1.5),
        use_gpu=False,
        seed=42,
    )
    skeleton, distance_field = generate_synthetic_fractal_tree(*params)

    assert skeleton.shape == distance_field.shape
    assert skeleton.dtype == "int"
    assert distance_field.dtype == "float32"
