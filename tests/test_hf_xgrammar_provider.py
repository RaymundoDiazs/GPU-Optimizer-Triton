from generation.hf_xgrammar_provider import call_hf_xgrammar
from generation.xgrammar_environment import verify_xgrammar_environment


def test_hf_xgrammar_provider_imports():
    assert callable(call_hf_xgrammar)


def test_xgrammar_environment_checker_exists():
    assert callable(verify_xgrammar_environment)
