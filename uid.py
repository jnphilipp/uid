#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2023 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# UID
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""UID."""


import joblib
import logging
import matplotlib.pyplot as plt
import sys

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawTextHelpFormatter,
)
from joblib import Parallel, delayed
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import List


__author__ = "J. Nathanael Philipp (jnphilipp)"
__contributors__ = ["Michael Richter"]
__copyright__ = "Copyright 2023 J. Nathanael Philipp (jnphilipp)"
__email__ = "nathanael@philipp.land"
__license__ = "GPLv3"
__version__ = "0.1.0"
__github__ = "https://github.com/jnphilipp/uid"


VERSION = (
    f"%(prog)s v{__version__}\n{__copyright__}\n"
    + "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>."
    + "\nThis is free software: you are free to change and redistribute it.\n"
    + "There is NO WARRANTY, to the extent permitted by law.\n\n"
    + f"Report bugs to {__github__}/issues."
    + f"\nWritten by {__author__} <{__email__}>\n"
    + f"Contributed by {', '.join(__contributors__)}"
)


class ArgFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Combination of ArgumentDefaultsHelpFormatter and RawTextHelpFormatter."""

    pass


def uid_from_file(
    path: str | Path,
    batch_size: int = 128,
    verbose: int = 10,
) -> List[float]:
    """Read surprisal values from file and calculate uid."""

    def calculate_uid(line: str) -> float:
        if ";" in line and "," in line:
            surprisal_values = [
                float(w.split("|")[1]) if "|" in w else 0.0
                for s in line.split(";")
                for w in s.split(",")
            ]
        elif "," in line:
            surprisal_values = [
                float(w.split("|")[1]) if "|" in w else 0.0 for w in line.split(",")
            ]

        sum_o2 = 0.0
        for i in range(1, len(surprisal_values) - 1):
            sum_o2 += (surprisal_values[i] - surprisal_values[i + 1]) ** 2

        return -1.0 * (sum_o2 / (len(surprisal_values) - 1))

    if isinstance(path, str):
        path = Path(path)

    with open(path, "r", encoding="utf8") as f:
        with Parallel(
            n_jobs=joblib.cpu_count(),
            verbose=verbose,
            batch_size=batch_size,
        ) as parallel:
            uids = parallel(delayed(calculate_uid)(line.strip()) for line in f)

    return uids


def filter_info(rec: logging.LogRecord) -> bool:
    """Log record filter for info and lower levels.

    Args:
     * rec: LogRecord object
    """
    return rec.levelno <= logging.INFO


if __name__ == "__main__":
    parser = ArgumentParser(prog="uid", formatter_class=ArgFormatter)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=VERSION,
    )
    parser.add_argument(
        "DATA",
        nargs="+",
        type=lambda p: Path(p).absolute(),
        help="file(s) to load surprisal data from.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help="name(s) to use in plot, needs to be the same length as DATA.",
    )

    # plot
    fill_group = parser.add_mutually_exclusive_group()
    fill_group.add_argument("--fill-area", action="store_true")
    fill_group.add_argument("--no-fill-area", action="store_false")
    parser.add_argument(
        "-o",
        "--output",
        type=lambda p: Path(p).absolute(),
    )

    # density
    parser.add_argument(
        "--x-start",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--bw-method",
        type=str,
        default="scott",
    )

    # logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbosity level; multiple times increases the level, the maximum is 3, "
        + "for debugging.",
    )
    parser.add_argument(
        "--log-format",
        default="%(message)s",
        help="set logging format.",
    )
    parser.add_argument(
        "--log-file",
        type=lambda p: Path(p).absolute(),
        help="log output to a file.",
    )
    parser.add_argument(
        "--log-file-format",
        default="[%(levelname)s] %(message)s",
        help="set logging format for log file.",
    )
    args = parser.parse_args()

    if args.verbose == 0:
        level = logging.WARNING
        verbosity = 0
    elif args.verbose == 1:
        level = logging.INFO
        verbosity = 1
    elif args.verbose == 2:
        level = logging.INFO
        verbosity = 10
    else:
        level = logging.DEBUG
        verbosity = 100

    handlers: List[logging.Handler] = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(filter_info)
    handlers.append(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    if "%(levelname)s" not in args.log_format:
        stderr_handler.setFormatter(
            logging.Formatter(f"[%(levelname)s] {args.log_format}")
        )
    handlers.append(stderr_handler)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(level)
        if args.log_file_format:
            file_handler.setFormatter(logging.Formatter(args.log_file_format))
        handlers.append(file_handler)

    logging.basicConfig(
        format=args.log_format,
        level=logging.DEBUG,
        handlers=handlers,
    )

    if args.names is not None and len(args.DATA) != len(args.names):
        logging.error("DATA and names needs to be the same length.")
        sys.exit(1)

    plt.figure(figsize=(25.6, 14.4), dpi=100)
    for i, path in enumerate(args.DATA):
        prob_density = gaussian_kde(
            uid_from_file(path, verbose=args.verbose),
            bw_method=float(args.bw_method)
            if args.bw_method.replace(".", "", 1).isdigit()
            else args.bw_method,
        )

        x = list(range(-args.x_start, 1))
        y = prob_density(x)
        plt.plot(x, y, label=args.names[i] if args.names else path.name)

        if args.fill_area and args.no_fill_area:
            plt.fill_between(x, y, alpha=0.4)
    plt.legend()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
