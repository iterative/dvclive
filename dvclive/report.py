import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dvclive.utils import _parse_tsv, is_cml_available


def build_report(plots_folder):
    from matplotlib import pyplot as plt
    from tabulate import tabulate

    if not is_cml_available():
        raise RuntimeError(
            "https://cml.dev/ is required in order to build the report."
        )

    report = ""
    history = {}
    plots = []
    headers = ["step"]
    with TemporaryDirectory() as tmpdir:
        for metric_file in Path(plots_folder).rglob("*.tsv"):
            metric_name = str(metric_file).replace(
                str(plots_folder) + os.path.sep, ""
            )
            metric_name = metric_name.replace(".tsv", "")

            data = _parse_tsv(metric_file)

            history[metric_name] = [x[metric_name] for x in data]
            headers.append(metric_name)

            plt.title(metric_name)
            plt.xlabel("step")
            plt.ylabel(metric_name)
            plt.plot(history[metric_name])
            plt.savefig(f"{tmpdir}/{metric_name}.png")
            plt.close()

            link = subprocess.run(
                ["cml", "publish", f"{tmpdir}/{metric_name}.png", "--md"],
                capture_output=True,
                text=True,
                check=True,
            )
            plots.append(str(link.stdout))

        table = tabulate(history, headers, showindex=True, tablefmt="github")
        report += f"## Metrics\n{table}"
        report += "\n## Plots"
        for link in plots:
            report += "\n---"
            report += f"\n{link}"

    return report
