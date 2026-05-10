from ojas.config import auto_configure
import os

try:
    config = auto_configure(workspace=os.path.abspath("."), ignore_ram=False)
    print("\n[SUCCESS] Configuration resolved.")
    print(f"Planner: {config.planner_model}")
    print(f"Coder: {config.coder_model}")
    print(f"Embed: {config.embed_model}")
except Exception as e:
    print(f"\n[ERROR] {e}")
