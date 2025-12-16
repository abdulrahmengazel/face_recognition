import traceback
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    import main as app
    app.start_gui()
except Exception:
    traceback.print_exc()
    raise
