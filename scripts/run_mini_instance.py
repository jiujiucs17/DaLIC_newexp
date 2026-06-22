import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_version",
        choices=("baseline", "dalic"),
        default="baseline",
        help="Select the mini-swe-agent prompt configuration.",
    )
    return parser.parse_args()


args = parse_args()

# read the text from prompts/task.txt
task = Path("/opt/dalic/prompts/task.txt").read_text()
config = (
    "mini.yaml"
    if args.agent_version == "baseline"
    else "/opt/dalic/prompts/dalic.yaml"
)

subprocess.run([
    "mini",
    "-y",
    "--exit-immediately",
    "--task", task,
    "-c", config,
    "-c", "agent.step_limit=100",
], check=True)