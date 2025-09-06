#!/usr/bin/env python3
"""
Test runner script for the Churn Copilot system.
"""

import subprocess
import sys
import os


def run_tests():
    """Run all unit tests."""
    print("🧪 Running Churn Copilot Unit Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🧪 Running {test_file}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            f"tests/{test_file}", 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], capture_output=False, text=True)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        success = run_specific_test(test_file)
    else:
        # Run all tests
        success = run_tests()
    
    if success:
        print("\n🎉 Testing completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
