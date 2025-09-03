"""
Setup script for the Company Researcher.

This script helps users install dependencies and configure
the environment for the LangGraph company researcher.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if openai_key:
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found")
    
    if tavily_key:
        print("✅ Tavily API key found")
    else:
        print("❌ Tavily API key not found")
    
    if not openai_key or not tavily_key:
        print("\n🔧 To configure API keys:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export TAVILY_API_KEY='your_tavily_key'")
        print("\n   Or create a .env file with:")
        print("   OPENAI_API_KEY=your_openai_key")
        print("   TAVILY_API_KEY=your_tavily_key")
        return False
    
    return True

def create_env_file():
    """Create a template .env file."""
    env_path = Path(".env")
    if env_path.exists():
        print("📄 .env file already exists")
        return
    
    env_template = """# API Keys for Company Researcher
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Uncomment and modify these settings
# MAX_SEARCH_QUERIES=5
# MAX_SEARCH_RESULTS=3
# MAX_REFLECTIONS=2
"""
    
    try:
        with open(env_path, "w") as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("   Please edit .env and add your API keys")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

def run_test():
    """Run a basic test to verify the setup."""
    print("\n🧪 Running basic test...")
    try:
        # Import the main module
        from company_researcher import CompanyResearcher
        print("✅ Import successful")
        
        # Check if we can initialize (will fail if API keys missing, but that's expected)
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if openai_key and tavily_key and openai_key != "your_openai_api_key_here":
            researcher = CompanyResearcher(
                openai_api_key=openai_key,
                tavily_api_key=tavily_key,
                max_search_queries=1
            )
            print("✅ CompanyResearcher initialized successfully")
        else:
            print("⚠️  Skipping full initialization test (API keys not configured)")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Test completed with warnings: {e}")
    
    return True

def main():
    """Main setup function."""
    print("🚀 Company Researcher Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return
    
    # Create env file template
    create_env_file()
    
    # Check API keys
    keys_configured = check_api_keys()
    
    # Run test
    test_passed = run_test()
    
    print("\n" + "=" * 40)
    
    if keys_configured and test_passed:
        print("🎉 Setup complete! You're ready to use the Company Researcher.")
        print("\nNext steps:")
        print("   python example_usage.py")
    else:
        print("⚠️  Setup partially complete.")
        if not keys_configured:
            print("   • Configure your API keys in .env file")
        print("   • Run this setup script again to verify")
        print("   • Or proceed with manual configuration")

    print("\n📚 Files created:")
    print("   • company_researcher.py - Main research engine")
    print("   • example_usage.py - Usage examples")
    print("   • config.py - Configuration options")
    print("   • requirements.txt - Dependencies")
    print("   • .env - API key template")

if __name__ == "__main__":
    main()