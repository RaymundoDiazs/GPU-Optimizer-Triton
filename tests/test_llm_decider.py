from models.llm_decider import decide_strategy


def test_decide_strategy_returns_expected_keys():
    decision = decide_strategy("output = torch.sum(x)", {}, "XGRAMMAR")

    assert set(decision) == {"strategy", "confidence", "explanation"}
    assert decision["strategy"] == "reduce_specialized"
    assert isinstance(decision["confidence"], float)
