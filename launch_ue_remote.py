"""
Remote Control API Approach: Launch UE and Execute Python via HTTP API

This approach:
1. Launches UE normally with Remote Control Web Server enabled
2. Waits for UE to start and Remote Control API to be ready
3. Sends Python script execution via HTTP POST request
4. UE stays open in user session mode - no auto-shutdown

Advantages:
- No additional libraries needed (uses standard urllib)
- Completely decoupled from CLI automation
- Stable and reliable HTTP communication
- Can execute multiple jobs without restarting UE

Requirements:
- Unreal Engine with Remote Control Web Interface plugin enabled
- Python 3.7+ (standard library only)
"""

import subprocess
import os
import sys
import time
import json
try:
    import urllib.request
    import urllib.error
except ImportError:
    print("ERROR: urllib not available")
    sys.exit(1)


class UnrealRemoteControl:
    """Handles launching UE and executing Python via Remote Control API"""

    def __init__(self, config):
        self.ue_editor = config.get('ue_editor')
        self.ue_project = config.get('ue_project')
        self.ue_python_script = config.get('ue_python_script')
        self.map_name = config.get('map_name', 'NewMap')
        self.remote_control_port = config.get('remote_control_port', 30010)
        self.remote_control_host = config.get('remote_control_host', 'localhost')

    def validate_paths(self):
        """Validate that all required paths exist"""
        if not os.path.exists(self.ue_editor):
            raise FileNotFoundError(f"UnrealEditor.exe not found at: {self.ue_editor}")

        if not os.path.exists(self.ue_project):
            raise FileNotFoundError(f"Project file not found at: {self.ue_project}")

        if not os.path.exists(self.ue_python_script):
            raise FileNotFoundError(f"Python script not found at: {self.ue_python_script}")

        print("✓ All paths validated successfully")

    def launch_ue_with_remote_control(self):
        """
        Launch Unreal Engine with Remote Control Web Server enabled.
        This keeps UE in normal user session mode.
        """
        print("=" * 70)
        print("  Launching Unreal Engine with Remote Control API")
        print("=" * 70)
        print()
        print(f"Unreal Editor:        {self.ue_editor}")
        print(f"Project:              {self.ue_project}")
        print(f"Map:                  {self.map_name}")
        print(f"Remote Control Port:  {self.remote_control_port}")
        print()

        # Build command with Remote Control enabled
        cmd = [
            self.ue_editor,
            self.ue_project,
            self.map_name,
            '-log',
            '-ExecCmds=WebControl.EnableServerOnStartup 1',  # Enable Remote Control
            f'-WebControlPort={self.remote_control_port}'     # Set port
        ]

        print("Command:")
        print(" ".join(cmd))
        print()
        print("=" * 70)
        print()

        print("Starting Unreal Engine with Remote Control...")
        print("Waiting for editor to fully load...")
        print()

        # Launch UE (non-blocking)
        subprocess.Popen(cmd)

        return True

    def wait_for_remote_control_ready(self, timeout=120):
        """
        Wait for Remote Control API to be ready by polling the health endpoint.
        Note: We check port 30000 (WebApp) for readiness, but will use port 30010 for API calls
        """
        print(f"Waiting for Remote Control API (timeout: {timeout}s)...")

        # Check WebApp port (30000) for readiness
        webapp_port = 30000
        base_url = f"http://{self.remote_control_host}:{webapp_port}/"

        start_time = time.time()
        check_count = 0

        while time.time() - start_time < timeout:
            try:
                # Try to connect to Remote Control API
                req = urllib.request.Request(base_url)
                response = urllib.request.urlopen(req, timeout=5)

                if response.status == 200:
                    print(f"✓ Remote Control API is ready at {base_url}")
                    return True

            except urllib.error.URLError as e:
                # API not ready yet
                if check_count % 10 == 0:  # Print error every 10 attempts
                    print(f"\n  Connection error: {e.reason if hasattr(e, 'reason') else e}")
                pass
            except Exception as e:
                # Other errors
                if check_count % 10 == 0:
                    print(f"\n  Error: {e}")
                pass

            check_count += 1

            time.sleep(2)
            print(".", end="", flush=True)

        raise TimeoutError(f"Remote Control API did not become ready within {timeout}s")

    def execute_python_script_via_remote_control(self):
        """
        Execute Python script via Remote Control API.
        Sends a command to execute the Python file using exec(open().read())
        """
        print()
        print("=" * 70)
        print("  Executing Python Script via Remote Control API")
        print("=" * 70)
        print()

        try:
            # Convert Windows path to forward slashes for Python
            script_path_normalized = os.path.abspath(self.ue_python_script).replace('\\', '/')

            print(f"Step 1: Preparing to execute Python script...")
            print(f"  Script path: {script_path_normalized}")

            # Build Remote Control API endpoint
            url = f"http://{self.remote_control_host}:{self.remote_control_port}/remote/object/call"

            # Build command that reads and executes the file
            # This avoids passing the entire script content as a string
            # Using exec(open().read()) tells Python to read and execute the file
            python_command = f"exec(open(r'{script_path_normalized}').read())"

            # Build the request payload
            payload = {
                "objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
                "functionName": "ExecutePythonCommand",
                "parameters": {
                    "PythonCommand": python_command
                },
                "generateTransaction": False
            }

            print("Step 2: Sending Python execution request to Remote Control API...")
            print(f"  Command: {python_command}")
            print()

            # Send PUT request (UE 5.6 requires PUT instead of POST)
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json'
                },
                method='PUT'
            )

            response = urllib.request.urlopen(req, timeout=30)
            response_data = response.read().decode('utf-8')

            # Parse response to check if execution succeeded
            try:
                response_json = json.loads(response_data)
                return_value = response_json.get('ReturnValue', False)

                print("✓ Python script execution request sent successfully!")
                print(f"  Return Value: {return_value}")
                print(f"  Full Response: {response_data[:300]}")
                print()

                if return_value:
                    print("✓ Python script is executing in UE!")
                else:
                    print("⚠ Script was sent but ReturnValue is False")
                    print("  Check UE Output Log for error details")

            except json.JSONDecodeError:
                print("✓ Request sent (could not parse response as JSON)")
                print(f"  Response: {response_data[:200]}")

            print()
            print("Monitor the UE Output Log window for:")
            print("  - Script execution progress")
            print("  - Movie Render Queue status")
            print("  - Any error messages")
            print()

            return True

        except urllib.error.HTTPError as e:
            print(f"✗ HTTP ERROR: {e.code} - {e.reason}")
            error_body = e.read().decode('utf-8')
            print(f"  Response: {error_body}")
            return False

        except Exception as e:
            print(f"✗ ERROR: Failed to execute via Remote Control: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        """Main execution flow"""
        try:
            # Step 1: Validate paths
            self.validate_paths()

            # Step 2: Launch UE with Remote Control
            if not self.launch_ue_with_remote_control():
                print("✗ ERROR: Failed to launch UE")
                return 1

            # Step 3: Wait for Remote Control API to be ready
            self.wait_for_remote_control_ready(timeout=120)

            # Step 4: Execute Python script via Remote Control API
            if not self.execute_python_script_via_remote_control():
                print("✗ ERROR: Failed to execute Python script")
                return 1

            print()
            print("=" * 70)
            print("✓ SUCCESS: Automation sequence complete!")
            print("=" * 70)
            print()
            print("Next steps:")
            print("1. Monitor UE's Output Log for script execution")
            print("2. UE will automatically close when rendering completes")
            print("3. Check output folder for rendered video")
            print()
            print("Note: UE will stay open. You can:")
            print("- Execute more scripts via this tool")
            print("- Manually close UE when done")
            print()

            return 0

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1


def load_config():
    """Load configuration"""
    # ============================================================================
    # CONFIGURATION - Update these paths to match your Windows machine
    # ============================================================================
    config = {
        'ue_editor': r'C:\Users\marketing\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe',
        'ue_project': r'C:\Users\marketing\Documents\Unreal Projects\male_runtime\MyMHProject.uproject',
        'ue_python_script': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ue_render_script.py'),
        'map_name': 'NewMap',
        'remote_control_port': 30010,
        'remote_control_host': 'localhost'
    }
    return config


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Launch UE and execute Python via Remote Control API'
    )

    parser.add_argument(
        '--ue-editor',
        type=str,
        help='Override path to UnrealEditor.exe'
    )

    parser.add_argument(
        '--ue-project',
        type=str,
        help='Override path to .uproject file'
    )

    parser.add_argument(
        '--python-script',
        type=str,
        help='Override path to Python script'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=30010,
        help='Remote Control API port (default: 30010)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Override with command line arguments
    if args.ue_editor:
        config['ue_editor'] = args.ue_editor

    if args.ue_project:
        config['ue_project'] = args.ue_project

    if args.python_script:
        config['ue_python_script'] = args.python_script

    if args.port:
        config['remote_control_port'] = args.port

    # Create automation instance
    automation = UnrealRemoteControl(config)

    # Run automation
    exit_code = automation.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
