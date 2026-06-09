from generation.tritonbench_constrained_decoding import (
    build_tritonbench_constrained_spec,
    select_tritonbench_grammar,
)


def test_selects_individual_tritonbench_grammar():
    grammar = select_tritonbench_grammar("tritonbench_t_016", mode="individual")

    assert "example 016" in grammar
    assert "add" in grammar


def test_selects_family_tritonbench_grammar():
    grammar = select_tritonbench_grammar("tritonbench_t_016", mode="family")

    assert "elementwise_kernel" in grammar
    assert "reduction_kernel" in grammar


def test_builds_constrained_spec_for_full_dataset():
    spec = build_tritonbench_constrained_spec("tritonbench_t_166", mode="individual")

    assert spec.task_id == "tritonbench_t_166"
    assert spec.grammar_mode == "individual"
    assert "Required wrapper" in spec.prompt
    assert "fftn" in spec.prompt
    assert spec.contract["source_index"] == 166


def test_constrained_prompt_separates_kernel_and_wrapper_rules():
    spec = build_tritonbench_constrained_spec("tritonbench_t_005", mode="family")

    assert "do not wrap the answer" in spec.prompt
    assert "do not use torch" in spec.prompt
    assert "tl.load(..., mask=mask)" in spec.prompt
    assert "plain Python wrapper" in spec.prompt
