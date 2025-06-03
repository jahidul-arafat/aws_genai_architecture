#!/usr/bin/env python3
"""
Script to build and upload package to TestPyPI
"""

import os
import sys
import subprocess
import shutil
import getpass

def run_command(cmd, description, interactive=False):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    if interactive:
        # For interactive commands, don't capture output
        result = subprocess.run(cmd)
        return result.returncode == 0
    else:
        # For non-interactive commands, capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error: {description} failed")
            print(f"Error output: {result.stderr}")
            return False
        else:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True

def upload_with_credentials():
    """Handle TestPyPI upload with proper credential handling."""
    print("\n🔐 TestPyPI Upload Credentials")
    print("=" * 40)
    print("📝 You need your TestPyPI API token")
    print("   Get it from: https://test.pypi.org/manage/account/token/")
    print("   Username: __token__")
    print("   Password: your-api-token (starts with 'pypi-')")

    # Option 1: Interactive upload (recommended)
    print("\n🚀 Upload Options:")
    print("1. Interactive upload (you'll be prompted for credentials)")
    print("2. Manual credential entry")
    print("3. Skip upload")

    choice = input("\nChoose option (1/2/3): ").strip()

    if choice == "1":
        print("\n🔄 Starting interactive upload...")
        print("💡 When prompted:")
        print("   Username: __token__")
        print("   Password: [paste your API token]")

        # Run twine upload interactively
        cmd = [sys.executable, '-m', 'twine', 'upload', '--repository', 'testpypi', 'dist/*']
        return run_command(cmd, "Uploading to TestPyPI (interactive)", interactive=True)

    elif choice == "2":
        # Manual credential entry
        username = "__token__"
        password = getpass.getpass("🔑 Enter your TestPyPI API token: ")

        if not password.startswith('pypi-'):
            print("⚠️  Warning: API token should start with 'pypi-'")
            continue_choice = input("Continue anyway? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                return False

        # Upload with credentials
        cmd = [
            sys.executable, '-m', 'twine', 'upload',
            '--repository', 'testpypi',
            '--username', username,
            '--password', password,
            'dist/*'
        ]

        # Don't show the command with password
        print(f"\n🔄 Uploading to TestPyPI with provided credentials...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Upload completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print("❌ Upload failed")
            print(f"Error: {result.stderr}")
            return False

    elif choice == "3":
        print("⏭️  Skipping upload")
        return True

    else:
        print("❌ Invalid choice")
        return False

def clean_egg_info():
    """Clean .egg-info directories properly."""
    import glob

    egg_dirs = glob.glob('*.egg-info')
    for egg_dir in egg_dirs:
        if os.path.exists(egg_dir):
            shutil.rmtree(egg_dir)
            print(f"   Removed: {egg_dir}")

def main():
    """Main build and upload process."""

    print("🚀 AWS GEenAI Simulation Package Build & Upload")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists('setup.py'):
        print("❌ Error: setup.py not found. Make sure you're in the package root directory.")
        return False

    # Clean previous builds
    print("\n🧹 Cleaning previous builds")
    dirs_to_clean = ['build', 'dist']

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed: {dir_name}")

    # Clean .egg-info directories
    clean_egg_info()

    # Install/upgrade build tools
    if not run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'build', 'twine'],
                       "Installing/upgrading build tools"):
        return False

    # Build the package
    if not run_command([sys.executable, '-m', 'build'],
                       "Building package"):
        return False

    # Check the built package
    if not run_command([sys.executable, '-m', 'twine', 'check', 'dist/*'],
                       "Checking package"):
        return False

    # Display build results
    print("\n📦 Package built successfully!")
    print("Files in dist/:")
    if os.path.exists('dist'):
        for file in os.listdir('dist'):
            size = os.path.getsize(os.path.join('dist', file)) / 1024  # KB
            print(f"   - {file} ({size:.1f} KB)")

    # Ask user if they want to upload
    upload_choice = input("\n🚀 Upload to TestPyPI? (y/n): ").strip().lower()

    if upload_choice in ['y', 'yes']:
        success = upload_with_credentials()

        if success:
            print("\n🎉 Package upload completed!")
            print("\n📦 To install from TestPyPI:")
            print("   pip install -i https://test.pypi.org/simple/ enhanced-dcgan-research")
            print("\n🔗 View your package at:")
            print("   https://test.pypi.org/project/enhanced-dcgan-research/")
            print("\n💡 Note: It may take a few minutes for the package to be available")
        else:
            print("\n❌ Upload failed. You can try uploading manually:")
            print("   python -m twine upload --repository testpypi dist/*")

    else:
        print("\n📦 Package built but not uploaded")
        print("\n💡 To upload later:")
        print("   python -m twine upload --repository testpypi dist/*")
        print("\n🔧 Or run this script again and choose to upload")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
        print("\n✅ Build process completed!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)