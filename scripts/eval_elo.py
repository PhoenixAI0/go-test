from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--katago-bin", required=True)
    parser.add_argument("--match-config", required=True)
    parser.add_argument("--log-file", default="match.log")
    parser.add_argument("--sgf-dir", default="match_sgfs")
    args = parser.parse_args()

    cmd = [
        args.katago_bin,
        "match",
        "-config",
        args.match_config,
        "-log-file",
        args.log_file,
        "-sgf-output-dir",
        args.sgf_dir,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
