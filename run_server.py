#!/usr/bin/env python
"""
Standalone script to run the Fall Detection API server
"""
import os
import sys
import uvicorn
import importlib.util
from pathlib import Path

def main():
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    
    # Add project root to path
    sys.path.append(str(project_root))
    
    # Import the app directly from the server module
    server_path = project_root / "v12" / "api" / "server.py"
    
    print(f"Starting Fall Detection API server from {server_path}")
    print("Server will be available at http://localhost:8000")
    
    # Run the server directly using the server.py file
    uvicorn.run("v12.api.server:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
