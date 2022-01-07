import json
from rich.console import Console
import re
c = Console()


def main():
    """ Load local data and clean it up. Saves to data/clean_lines.txt"""
    # ------------- Loading -----------------
    with open("data/all_scripts_raw.json", "r") as f:
        data = json.load(f)

    series_data = data["TNG"]

    # ------------- Cleaning -----------------
    all_lines = []
    for _, script in series_data.items():
        lines = script.split(" \n")  # space is important!
        lines = [
            re.sub(r"\s+", " ", l.replace("\n", " ").replace("\xa0", " "))
            for l in lines
            if l != "" and not "trademark" in l.lower()
        ]
        all_lines.extend(lines)

    # for idx, line in enumerate(all_lines):
    #     if idx == len(all_lines) - 1:
    #         break  # last line

    #     all_lines[idx] = line + " " + all_lines[idx + 1].split()[0]

    # merge lines
    n_to_merge = 3
    merged_lines = []
    for idx in range(0, len(all_lines) - n_to_merge, n_to_merge):
        merged_lines.append(
            " ".join(all_lines[idx : idx + n_to_merge])
        )

    # ------------- Writing output -----------------
    with open("data/clean_lines.txt", "w") as f:
        for line in merged_lines[1:]:  # first line is credits to data source
            f.write(line + "\n")

    c.print("[bold green][INFO][/] Cleaned up data and saved to data/clean_lines.txt.")


if __name__ == "__main__":
    main()
