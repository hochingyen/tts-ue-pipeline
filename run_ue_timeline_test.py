#!/usr/bin/env python3
"""
Launch UE and run the timeline adjustment test.

This will:
1. Launch Unreal Engine
2. Wait for Remote Control API to be ready
3. Send the test script to UE
4. UE will test timeline adjustment and log results
5. Keep UE open so you can inspect the timeline in the editor
"""

import sys
import os
import subprocess
import time
import requests

# UE configuration (Windows paths)
UE_EDITOR_PATH = r"C:\Users\marketing\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
UE_PROJECT_PATH = r"C:\Users\marketing\Documents\Unreal Projects\male_runtime\MyMHProject.uproject"
REMOTE_CONTROL_PORT = 30010
HEALTH_CHECK_PORT = 30000

def launch_ue():
    """Launch Unreal Engine with Remote Control enabled."""
    print("=" * 80)
    print("Launching Unreal Engine for timeline test...")
    print("=" * 80)

    if not os.path.exists(UE_EDITOR_PATH):
        print(f"ERROR: UE editor not found: {UE_EDITOR_PATH}")
        return None

    if not os.path.exists(UE_PROJECT_PATH):
        print(f"ERROR: UE project not found: {UE_PROJECT_PATH}")
        return None

    cmd = [
        UE_EDITOR_PATH,
        UE_PROJECT_PATH,
        "-RemoteControlIsHeadless",
        "-RCWebInterfaceEnable",
        "-RCWebInterfacePort=30010"
    ]

    print(f"Command: {' '.join(cmd)}")
    print("\nStarting UE...")

    process = subprocess.Popen(cmd)
    print(f"✓ UE process started (PID: {process.pid})")

    return process

def wait_for_remote_control(timeout=120):
    """Wait for Remote Control API to be ready."""
    print("\nWaiting for Remote Control API to be ready...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{HEALTH_CHECK_PORT}")
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"✓ Remote Control API is ready! (took {elapsed:.1f}s)")
                return True
        except requests.exceptions.ConnectionError:
            pass

        elapsed = time.time() - start_time
        print(f"  Waiting... ({elapsed:.0f}s / {timeout}s)", end='\r')
        time.sleep(1)

    print(f"\n✗ Timeout waiting for Remote Control API ({timeout}s)")
    return False

def send_test_script():
    """Send the timeline test script to UE."""
    print("\n" + "=" * 80)
    print("Sending timeline test script to UE...")
    print("=" * 80)

    # Read test script
    script_path = "test_ue_timeline.py"
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Send to UE via Remote Control API
    url = f"http://localhost:{REMOTE_CONTROL_PORT}/remote/object/call"

    payload = {
        "objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
        "functionName": "ExecutePythonCommand",
        "parameters": {
            "PythonCommand": script_content
        },
        "generateTransaction": False
    }

    try:
        response = requests.put(url, json=payload)
        if response.status_code == 200:
            print("✓ Test script sent to UE successfully!")
            print("\nCheck the UE Output Log for results:")
            print("  Window > Developer Tools > Output Log")
            print("\nLook for:")
            print("  ✅ SUCCESS! Timeline was adjusted correctly!")
            print("  or")
            print("  ❌ FAILED! Timeline was NOT adjusted correctly!")
            return True
        else:
            print(f"✗ Failed to send script: HTTP {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error sending script: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("UE TIMELINE ADJUSTMENT TEST")
    print("=" * 80)

    print("\nBEFORE RUNNING:")
    print("  1. Edit test_ue_timeline.py")
    print("  2. Set AUDIO_FILE_PATH to your actual WAV file path")
    print("  3. Save the file")

    input("\nPress Enter when ready to launch UE...")

    # Launch UE
    process = launch_ue()
    if not process:
        sys.exit(1)

    # Wait for Remote Control API
    if not wait_for_remote_control():
        print("\nTIP: Make sure UE Remote Control plugin is enabled")
        print("     Project Settings > Plugins > Remote Control")
        process.terminate()
        sys.exit(1)

    # Send test script
    if not send_test_script():
        process.terminate()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Test complete! UE will remain open.")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check UE Output Log for test results")
    print("  2. Open the Level Sequence in UE")
    print("  3. Check if the timeline was adjusted to match your audio duration")
    print("  4. Close UE when done")

    # Keep script running
    input("\nPress Enter to close UE...")
    process.terminate()

if __name__ == "__main__":
    main()
