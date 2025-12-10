"""
Run all tests in chronological order (oldest to newest)

ลำดับการทดสอบ:
1. Embedding Quality Inspector (เดิม: Embedding Physics Inspector) - เก่าที่สุด
2. Baseline Behavioral Analyzer - กลาง
3. EIMAS Analyzer - ใหม่ที่สุด
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def print_header(title, version=""):
    """Print test section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    if version:
        print(f"  Version: {version}")
    print("="*80)


def run_embedding_quality_tests():
    """Run tests for Embedding Quality Inspector (oldest)"""
    print_header("1. EMBEDDING QUALITY INSPECTOR", "Original Implementation")
    
    try:
        from quick_test import quick_test
        print("\nRunning quick_test.py...")
        good_result, bad_result = quick_test()
        print("✅ Embedding Quality Inspector tests completed")
        return True
    except Exception as e:
        print(f"❌ Embedding Quality Inspector tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_behavioral_analyzer_tests():
    """Run tests for Baseline Behavioral Analyzer (middle)"""
    print_header("2. BASELINE BEHAVIORAL ANALYZER", "v0.1")
    
    try:
        # Quick test
        print("\nRunning quick_test_behavioral.py...")
        from quick_test_behavioral import quick_test
        analyzer, results = quick_test()
        print("✅ Quick test completed")
        
        # Full test suite
        print("\nRunning test_behavioral_analyzer.py...")
        from test_behavioral_analyzer import run_all_tests as run_behavioral_tests
        run_behavioral_tests()
        print("✅ Baseline Behavioral Analyzer tests completed")
        return True
    except Exception as e:
        print(f"❌ Baseline Behavioral Analyzer tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_eimas_tests():
    """Run tests for EIMAS Analyzer (newest)"""
    print_header("3. EIMAS ANALYZER", "v1.0")
    
    try:
        # Quick demo
        print("\nRunning eimas_analyzer.py demo...")
        from cogman_tools.eimas_analyzer  import demo_eimas
        analyzer, results = demo_eimas()
        print("✅ Demo completed")
        
        # Full test suite
        print("\nRunning test_eimas_analyzer.py...")
        from test_eimas_analyzer import run_all_tests as run_eimas_tests
        passed, failed = run_eimas_tests()
        print("✅ EIMAS Analyzer tests completed")
        return failed == 0
    except Exception as e:
        print(f"❌ EIMAS Analyzer tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests in chronological order"""
    print("\n" + "="*80)
    print("  COGMAN TOOLS - COMPREHENSIVE TEST SUITE")
    print("  Running all tests in chronological order (oldest to newest)")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}")
    
    results = {
        'embedding_quality': False,
        'behavioral_analyzer': False,
        'eimas_analyzer': False
    }
    
    # 1. Embedding Quality Inspector (oldest)
    results['embedding_quality'] = run_embedding_quality_tests()
    
    # 2. Baseline Behavioral Analyzer (middle)
    results['behavioral_analyzer'] = run_behavioral_analyzer_tests()
    
    # 3. EIMAS Analyzer (newest)
    results['eimas_analyzer'] = run_eimas_tests()
    
    # Summary
    print("\n" + "="*80)
    print("  FINAL TEST SUMMARY")
    print("="*80)
    print(f"Completed at: {datetime.now().isoformat()}")
    print("\nResults:")
    print(f"  1. Embedding Quality Inspector: {'✅ PASSED' if results['embedding_quality'] else '❌ FAILED'}")
    print(f"  2. Baseline Behavioral Analyzer: {'✅ PASSED' if results['behavioral_analyzer'] else '❌ FAILED'}")
    print(f"  3. EIMAS Analyzer: {'✅ PASSED' if results['eimas_analyzer'] else '❌ FAILED'}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TEST SUITES PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

