from generation.hf_xgrammar_provider import call_hf_xgrammar


def test_hf_xgrammar_provider_imports():
    assert callable(call_hf_xgrammar)