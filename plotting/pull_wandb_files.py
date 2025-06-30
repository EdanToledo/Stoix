#!/usr/bin/env python3

import argparse
import os
import zipfile
from pathlib import Path

import wandb


def parse_args():
    """
    Parses command-line arguments for this script.
    """
    parser = argparse.ArgumentParser(
        description="""Download and optionally unzip files from W&B runs based on given filters.
        This is mainly a helper function to pull the JSON results files from different wandb runs
        logged when using stoix systems."""
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="",
        help="W&B entity (user or organization) from which to download.",
    )
    parser.add_argument(
        "--project", type=str, default="", help="Name of the W&B project from which to download."
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["stoix"],
        help=(
            "One or more tags used to filter runs (logical OR). "
            "For example, '--tags tag1 tag2' matches runs that have tag1 OR tag2."
        ),
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="metrics.zip",
        help="Name of the file to look for and download from each run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unzipped_files",
        help="Directory where downloaded files (and unzipped contents) will be stored.",
    )
    parser.add_argument(
        "--finished_only",
        action="store_true",
        help="If set, only download from runs that have finished (state=finished).",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point: queries W&B for runs with specified tags, downloads the desired file,
    and (if it's a zip) unzips it into a dedicated subdirectory.

    :return: Exit code (0 if everything succeeds).
    """
    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B API
    api = wandb.Api()

    # Build filters for runs based on tags and possibly state
    filters = {"tags": {"$in": args.tags}}
    if args.finished_only:
        filters["state"] = "finished"

    print(f"Querying runs from '{args.entity}/{args.project}' with filters: {filters}")
    runs = api.runs(path=f"{args.entity}/{args.project}", filters=filters)

    if not runs:
        print("No runs found with the given filters.")
        return 0

    # Loop over each run that matches the filters
    for run in runs:
        run_name = run.name or run.id  # Some runs may not have a name
        run_id = run.id
        print(f"Processing run '{run_name}' (ID: {run_id})")

        # Check if the desired file is in this run
        desired_file = None
        for wandb_file in run.files():
            # We do a simple substring check here:
            if args.filename in wandb_file.name:
                desired_file = wandb_file
                break

        if not desired_file:
            print(f" - File '{args.filename}' NOT FOUND in this run. Skipping.\n")
            continue

        # Create a unique subdirectory for this run.
        # Combining run.id and run.name ensures uniqueness, even if names repeat.
        safe_run_name = run_name.replace(" ", "_")
        run_output_dir = Path(args.output_dir) / f"{run_id}_{safe_run_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Download the file into the run's subdirectory
        print(f" - Downloading '{args.filename}' to '{run_output_dir}'...")
        downloaded_path = desired_file.download(root=str(run_output_dir), replace=True)

        # Check if it's a ZIP
        if not downloaded_path.name.endswith(".zip"):
            print(
                f" - The downloaded file '{downloaded_path.name}' is not a ZIP. Skipping unzip.\n"
            )
            continue

        # Unzip the file into the same subdirectory
        print(f" - Unzipping '{downloaded_path.name}' in '{run_output_dir}'...")
        try:
            # Resolve the full path to avoid confusion in relative directories
            zip_full_path = Path(downloaded_path.name).resolve()
            with zipfile.ZipFile(zip_full_path, "r") as zip_ref:
                zip_ref.extractall(run_output_dir)
            print(" - Extraction complete.\n")
        except zipfile.BadZipFile:
            print(f" - ERROR: '{downloaded_path.name}' is not a valid zip file.\n")

    print("All matching runs processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
