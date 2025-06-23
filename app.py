#!/usr/bin/env python3
"""
Debug startup script to diagnose container issues
Run this instead of your main app to debug what's happening
"""

import os
import sys
import subprocess
import pkg_resources
import importlib
import traceback
from pathlib import Path

def print_section(title):
    print("="*60)
    print(f" {title}")
    print("="*60)

def check_python_environment():
    print_section("PYTHON ENVIRONMENT")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PATH environment variable: {os.environ.get('PATH', 'NOT SET')}")

def check_installed_packages():
    print_section("INSTALLED PACKAGES")
    try:
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        print(f"Total packages installed: {len(installed_packages)}")
        
        # Check for key packages
        key_packages = ['flask', 'gunicorn', 'langchain', 'chromadb', 'openai', 'boto3']
        for package in key_packages:
            if package in installed_packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    print(f"✅ {package}: {version}")
                except:
                    print(f"❌ {package}: FOUND BUT VERSION UNKNOWN")
            else:
                print(f"❌ {package}: NOT FOUND")
                
        print("\nAll installed packages:")
        for package in sorted(installed_packages):
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"  {package}: {version}")
            except:
                print(f"  {package}: VERSION UNKNOWN")
                
    except Exception as e:
        print(f"Error checking packages: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def check_gunicorn():
    print_section("GUNICORN CHECK")
    try:
        # Check if gunicorn is available
        result = subprocess.run(['which', 'gunicorn'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Gunicorn found at: {result.stdout.strip()}")
        else:
            print("❌ Gunicorn not found in PATH")
        
        # Try to import gunicorn
        try:
            import gunicorn
            print(f"✅ Gunicorn can be imported, version: {gunicorn.__version__}")
        except ImportError as e:
            print(f"❌ Cannot import gunicorn: {e}")
        
        # Try to run gunicorn --version
        try:
            result = subprocess.run(['gunicorn', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Gunicorn version command output: {result.stdout.strip()}")
            else:
                print(f"❌ Gunicorn version command failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ Gunicorn command not found")
        except Exception as e:
            print(f"❌ Error running gunicorn --version: {e}")
            
    except Exception as e:
        print(f"Error checking gunicorn: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def check_environment_variables():
    print_section("ENVIRONMENT VARIABLES")
    important_vars = [
        'OPENAI_API_KEY', 'S3_BUCKET_NAME', 'S3_CHROMA_KEY', 
        'AWS_REGION', 'CHROMA_PATH', 'FLASK_SECRET_KEY',
        'PORT', 'FLASK_ENV'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'SECRET' in var:
                print(f"✅ {var}: {'*' * min(len(value), 20)} (SET)")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")

def check_file_system():
    print_section("FILE SYSTEM")
    current_dir = Path(os.getcwd())
    print(f"Current directory: {current_dir}")
    print(f"Directory contents:")
    
    try:
        for item in current_dir.iterdir():
            if item.is_file():
                print(f"  📄 {item.name} ({item.stat().st_size} bytes)")
            elif item.is_dir():
                print(f"  📁 {item.name}/")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Check specific files
    important_files = ['app.py', 'requirements.txt', 'Dockerfile']
    for filename in important_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"✅ {filename} exists ({filepath.stat().st_size} bytes)")
        else:
            print(f"❌ {filename} not found")

def test_imports():
    print_section("IMPORT TESTS")
    test_modules = [
        'flask', 'gunicorn', 'langchain', 'chromadb', 
        'openai', 'boto3', 'sentence_transformers'
    ]
    
    for module in test_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}: SUCCESS")
        except ImportError as e:
            print(f"❌ {module}: FAILED - {e}")
        except Exception as e:
            print(f"⚠️  {module}: ERROR - {e}")

def test_flask_app():
    print_section("FLASK APP TEST")
    try:
        # Try to import the app
        sys.path.append(os.getcwd())
        
        print("Attempting to import app...")
        import app as flask_app
        print("✅ App module imported successfully")
        
        print("Checking if app object exists...")
        if hasattr(flask_app, 'app'):
            print("✅ App object found")
            
            # Try to create test client
            print("Creating test client...")
            with flask_app.app.test_client() as client:
                print("✅ Test client created successfully")
                
                # Try a simple request
                print("Testing / endpoint...")
                response = client.get('/')
                print(f"✅ / endpoint responded with status: {response.status_code}")
                
        else:
            print("❌ App object not found in module")
            
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def main():
    print_section("CONTAINER DEBUG REPORT")
    print(f"Starting debug session at: {os.getcwd()}")
    
    check_python_environment()
    check_environment_variables()
    check_file_system()
    check_installed_packages()
    check_gunicorn()
    test_imports()
    test_flask_app()
    
    print_section("DEBUG COMPLETE")
    print("If gunicorn is available, try running:")
    print("  gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 300 app:app")
    print("\nIf gunicorn is not available, try running with Python directly:")
    print("  python app.py")

if __name__ == "__main__":
    main()
