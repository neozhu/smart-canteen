#!/usr/bin/env python3
"""
Build script for Smart-Canteen backend
Creates a standalone executable using PyInstaller
"""
import os
import shutil
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    print("✅ Clean complete")

def build_frontend():
    """Build Next.js frontend as static export"""
    print("🏗️  Building frontend...")
    os.chdir(FRONTEND_DIR)
    
    # Install dependencies
    subprocess.run(["npm", "install"], check=True)
    
    # Build static export
    subprocess.run(["npm", "run", "build"], check=True)
    
    print("✅ Frontend build complete")
    os.chdir(ROOT_DIR)

def build_backend():
    """Build backend as standalone executable"""
    print("🏗️  Building backend executable...")
    
    # Create spec file for PyInstaller
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['backend/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend/data', 'data'),
        ('backend/models', 'models'),
        ('frontend/out', 'frontend/out'),
    ],
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='smart-canteen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='infra/icon.ico' if os.path.exists('infra/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='smart-canteen',
)
"""
    
    spec_file = ROOT_DIR / "smart-canteen.spec"
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    # Run PyInstaller
    subprocess.run([
        "pyinstaller",
        "--clean",
        str(spec_file)
    ], check=True)
    
    print("✅ Backend build complete")

def create_installer():
    """Create installer package"""
    print("📦 Creating installer package...")
    
    installer_dir = DIST_DIR / "installer"
    installer_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy executable
    exe_source = DIST_DIR / "smart-canteen"
    if exe_source.exists():
        shutil.copytree(exe_source, installer_dir / "smart-canteen")
    
    # Create install script
    install_script = """
@echo off
echo Installing Smart-Canteen...

REM Create installation directory
set INSTALL_DIR=%ProgramFiles%\\SmartCanteen
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy files
xcopy /E /I /Y smart-canteen "%INSTALL_DIR%"

REM Create desktop shortcut
set SCRIPT_DIR=%~dp0
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%USERPROFILE%\\Desktop\\Smart Canteen.lnk'); $SC.TargetPath = '%INSTALL_DIR%\\smart-canteen.exe'; $SC.Save()"

REM Register service (optional)
echo To run as service, execute: sc create SmartCanteen binPath= "%INSTALL_DIR%\\smart-canteen.exe"

echo Installation complete!
pause
"""
    
    with open(installer_dir / "install.bat", 'w') as f:
        f.write(install_script)
    
    print("✅ Installer created at:", installer_dir)

def main():
    """Main build process"""
    print("🚀 Starting Smart-Canteen build process...")
    
    try:
        clean_build()
        build_frontend()
        build_backend()
        create_installer()
        
        print("\n✨ Build completed successfully!")
        print(f"📁 Output directory: {DIST_DIR}")
        print(f"📦 Installer: {DIST_DIR / 'installer'}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
