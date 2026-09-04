import argparse
import os
from pathlib import Path
import subprocess

import yaml

CONTEXT_WINDOW_MODEL_CLASS = "minisweagent.models.context_window_model.ContextWindowModel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_version",
        choices=("baseline", "dalic", "original"),
        default="baseline",
        help="Select the mini-swe-agent prompt configuration.",
    )
    return parser.parse_args()


def should_use_context_window_model(config: str, model_name: str) -> bool:
    model_config = yaml.safe_load(Path(config).read_text()).get("model", {})
    model_family = model_name.rsplit("/", 1)[-1].lower()
    return (
        model_config.get("context_length") == 32768
        and model_family.startswith(("deepseek", "gpt"))
    )


args = parse_args()

# read the text from prompts/task.txt
task = Path("/opt/dalic/prompts/task.txt").read_text()
if args.agent_version == "original":
    config = "mini.yaml"
elif args.agent_version == "dalic":
    config = "/opt/dalic/prompts/dalic_v1.yaml"
else:
    config = "/opt/dalic/prompts/baseline.yaml"

command = [
    "mini",
    "-y",
    "--exit-immediately",
    "--task", task,
    "-c", config,
    "-c", "agent.step_limit=50",
]
model_name = os.getenv("MSWEA_MODEL_NAME", "")
use_context_window_model = (
    args.agent_version != "original"
    and should_use_context_window_model(config, model_name)
)
if use_context_window_model and model_name.rsplit("/", 1)[-1].lower().startswith("gpt"):
    command.extend(["-c", "model.model_kwargs.tool_choice=required"])

if use_context_window_model:
    command.extend(["-c", f"model.model_class={CONTEXT_WINDOW_MODEL_CLASS}"])

subprocess.run(command, check=True)
