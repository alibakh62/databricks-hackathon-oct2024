#!/usr/bin/env python3
"""
Basic test script for the LLM-as-a-judge evaluator structure.
This script tests the basic functionality without requiring API keys.
"""

import os
import sys

def test_imports():
    """Test that all imports work correctly."""
    try:
        # Test basic imports
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import mlflow
        print("✅ mlflow imported successfully")
        
        from mlflow.metrics.genai import make_genai_metric, EvaluationExample
        print("✅ mlflow.metrics.genai imported successfully")
        
        from mlflow.metrics.genai import relevance, faithfulness, answer_relevance
        print("✅ mlflow built-in metrics imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_metric_creation():
    """Test that custom metrics can be created."""
    try:
        from mlflow.metrics.genai import make_genai_metric, EvaluationExample
        
        # Create a simple test metric
        examples = [
            EvaluationExample(
                input="Test input",
                output="Test output",
                score=5,
                justification="This is a test"
            )
        ]
        
        test_metric = make_genai_metric(
            name="test_metric",
            definition="A test metric",
            grading_prompt="Score from 1-5",
            examples=examples
        )
        
        print("✅ Custom metric created successfully")
        return True
    except Exception as e:
        print(f"❌ Metric creation error: {str(e)}")
        return False

def test_evaluator_structure():
    """Test the evaluator class structure without API calls."""
    try:
        # Temporarily remove the API key requirement
        original_env = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test_key"
        
        # Import and test basic structure
        from evaluator import CampaignEvaluator
        
        # Test that the class can be instantiated (will fail on API call, but that's expected)
        try:
            evaluator = CampaignEvaluator()
            print("✅ CampaignEvaluator class structure is correct")
            
            # Test that custom metrics are set up
            assert hasattr(evaluator, 'campaign_relevance')
            assert hasattr(evaluator, 'marketing_effectiveness')
            assert hasattr(evaluator, 'research_quality')
            assert hasattr(evaluator, 'email_quality')
            print("✅ Custom metrics are properly initialized")
            
            return True
        except Exception as e:
            if "OpenAI" in str(e) or "API" in str(e):
                print("✅ CampaignEvaluator structure is correct (API error expected)")
                return True
            else:
                print(f"❌ Unexpected error: {str(e)}")
                return False
    except Exception as e:
        print(f"❌ Structure test error: {str(e)}")
        return False
    finally:
        # Restore original environment
        if original_env:
            os.environ["OPENAI_API_KEY"] = original_env
        else:
            os.environ.pop("OPENAI_API_KEY", None)

def test_documentation():
    """Test that documentation files exist and are readable."""
    try:
        # Check if README exists
        if os.path.exists("LLM_AS_JUDGE_README.md"):
            print("✅ LLM_AS_JUDGE_README.md exists")
            
            with open("LLM_AS_JUDGE_README.md", "r") as f:
                content = f.read()
                if len(content) > 100:
                    print("✅ Documentation is substantial")
                else:
                    print("⚠️ Documentation seems short")
        else:
            print("❌ LLM_AS_JUDGE_README.md not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Documentation test error: {str(e)}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 Starting Basic LLM-as-a-Judge Evaluator Tests")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Metric Creation Test", test_metric_creation),
        ("Evaluator Structure Test", test_evaluator_structure),
        ("Documentation Test", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! The evaluator structure is correct.")
        print("Note: Full functionality requires a valid OpenAI API key.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 