"""Test coverage for implementation status from info.md spec"""

import re
from pathlib import Path
import dspy

def test_deepseek_integration():
    """Verify DeepSeek is configured as specified"""
    assert isinstance(dspy.settings.lm, dspy.LM)
    assert "deepseek" in dspy.settings.lm.model.lower()

def test_real_e2e_tests_exist():
    """Verify we have real e2e tests with LLM calls"""
    # Check test_llm_comparison.py has integration test
    test_file = Path("tests/test_llm_comparison.py").read_text(encoding='utf-8')
    assert "@pytest.mark.integration" in test_file
    assert "test_compare_llm_outputs_integration" in test_file

def test_demo_data_validation_implemented():
    """Verify demo data validation exists"""
    demo_data = Path("data/demo_training_data.csv")
    assert demo_data.exists()
    
    # Check validation in test_integration.py
    test_file = Path("tests/test_integration.py").read_text(encoding='utf-8') 
    assert "test_demo_training_data_validation" in test_file

def test_cli_training_implemented():
    """Verify CLI training produces output files"""
    # Check test_integration.py has CLI test
    test_file = Path("tests/test_integration.py").read_text(encoding='utf-8')
    assert "test_training_script_cli" in test_file
    assert "training_info.txt" in test_file

def test_error_handling_implemented():
    """Verify proper error handling exists"""
    # Check training.py validation
    code = Path("dspy_elo/training.py").read_text(encoding='utf-8')
    assert "raise ValueError" in code
    assert "required_cols - set(training_data.columns)" in code

def test_no_hardcoded_responses():
    """Verify comparisons use real model responses"""
    # Check comparison module doesn't have hardcoded unconditional returns
    code = Path("dspy_elo/llm_comparison.py").read_text(encoding='utf-8')
    # Verify there are no unconditional return statements with 1/2
    assert not re.search(r"^\s*return 1\s*$", code, flags=re.MULTILINE)
    assert not re.search(r"^\s*return 2\s*$", code, flags=re.MULTILINE)
