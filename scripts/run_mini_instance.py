from pathlib import Path
import os
import subprocess

# read the text from prompts/task.txt
task = Path("/opt/dalic/prompts/task.txt").read_text()

subprocess.run([
    "mini",
    "-y",
    "--exit-immediately",
    "--task", task,
    "-c", "mini.yaml",
    "-c", "agent.step_limit=100",
], check=True)