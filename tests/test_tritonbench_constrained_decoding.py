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

    assert "elementwise_body" in grammar
    assert "reduction_body" in grammar


def test_selects_universal_tritonbench_grammar():
    grammar = select_tritonbench_grammar("tritonbench_t_016", mode="universal")

    assert "root ::= module" in grammar
    assert "required_parallelism" in grammar
    assert "required_memory_io" in grammar
    assert "@triton.jit" in grammar


def test_builds_constrained_spec_for_full_dataset():
    spec = build_tritonbench_constrained_spec("tritonbench_t_166", mode="individual")

    assert spec.task_id == "tritonbench_t_166"
    assert spec.grammar_mode == "individual"
    assert "Required wrapper" in spec.prompt
    assert "fftn" in spec.prompt
    assert spec.contract["source_index"] == 166


def test_builds_universal_constrained_spec():
    spec = build_tritonbench_constrained_spec("tritonbench_t_016", mode="universal")

    assert spec.task_id == "tritonbench_t_016"
    assert spec.grammar_mode == "universal"
    assert "Grammar mode: universal" in spec.prompt
    assert "required_parallelism" in spec.grammar_text
