import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from browser_use.llm.azure_foundry.chat import ChatAzureFoundry
    print("Successfully imported ChatAzureFoundry from module")
    
    from browser_use.llm import ChatAzureFoundry as ChatAzureFoundryFromInit
    print("Successfully imported ChatAzureFoundry from init")
    
    from browser_use.llm.models import ChatAzureFoundry as ChatAzureFoundryFromModels
    print("Successfully imported ChatAzureFoundry from models")
    
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
