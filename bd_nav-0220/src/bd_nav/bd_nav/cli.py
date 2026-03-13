import os
import sys

from ament_index_python.packages import get_package_share_directory

def main():
    try:
        share = get_package_share_directory('bd_nav')
    except Exception as e:
        print(f"[bd_nav] Failed to look up share directory: {e}", file=sys.stderr)
        sys.exit(1)

    app = os.path.join(share, 'ui', 'user_input.py')  # path that will be installed by setup.py
    # Replace with streamlit execution (exec streamlit in the current process)
    os.execvp('streamlit', ['streamlit', 'run', app, '--server.address=0.0.0.0'])
