from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    alembic_ini = Path(__file__).parent.parent / "alembic.ini"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(alembic_ini), "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Alembic migration failed:\n{result.stderr}")
        sys.exit(1)
    print("Database migrated successfully.")


if __name__ == "__main__":
    main()
