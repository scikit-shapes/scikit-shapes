"""Command-line interface of the registration, and compatibility module.

The code lives in the optimal_steps package; `from register import register,
RegistrationConfig` keeps working.

    python register.py --help
"""

from optimal_steps import *  # noqa: F403
from optimal_steps.cli import main

if __name__ == "__main__":
    main()
