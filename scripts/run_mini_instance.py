import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_version",
        choices=("baseline", "dalic", "original"),
        default="baseline",
        help="Select the mini-swe-agent prompt configuration.",
    )
    return parser.parse_args()


args = parse_args()

# read the text from prompts/task.txt
task = Path("/opt/dalic/prompts/task.txt").read_text()
if args.agent_version == "original":
    config = "mini.yaml"
elif args.agent_version == "dalic":
    config = "/opt/dalic/prompts/dalic_v2.yaml"
else:
    config = "/opt/dalic/prompts/baseline.yaml"

subprocess.run([
    "mini",
    "-y",
    "--exit-immediately",
    "--task", task,
    "-c", config,
    "-c", "agent.step_limit=50",
], check=True)