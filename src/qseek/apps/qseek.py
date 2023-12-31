#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from pathlib import Path

import nest_asyncio
from pkg_resources import get_distribution

nest_asyncio.apply()

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    prog="qseek",
    description="qseek - The wholesome earthquake detector 🚀",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="count",
    default=0,
    help="increase verbosity of the log messages, repeat to increase. "
    "Default level is INFO",
)
parser.add_argument(
    "--version",
    action="version",
    version=get_distribution("qseek").version,
    help="show version and exit",
)

subparsers = parser.add_subparsers(
    title="commands",
    required=True,
    dest="command",
    description="Available commands to run qseek. Get command help with "
    "`qseek <command> --help`.",
)

subparsers.add_parser(
    "config",
    help="print a new config",
    description="initialze a new default config configuration file.",
)

run = subparsers.add_parser(
    "search",
    help="start a search",
    description="detect, localize and characterize earthquakes in a dataset",
)
config_arg = run.add_argument(
    "config",
    type=Path,
    help="path to config file",
)
run.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="backup old rundir and create a new",
)

continue_run = subparsers.add_parser(
    "continue",
    help="continue an aborted run",
    description="continue a run from an existing rundir",
)
rundir_continue = continue_run.add_argument(
    "rundir",
    type=Path,
    help="existing runding to continue",
)

features_extract = subparsers.add_parser(
    "feature-extraction",
    help="extract features from an existing run",
    description="modify the search.json for re-evaluation of the event's features",
)
rundir_features = features_extract.add_argument(
    "rundir",
    type=Path,
    help="path of existing run",
)

station_corrections = subparsers.add_parser(
    "corrections",
    help="analyse station corrections from existing run",
    description="analyze and plot station corrections from a finished run",
)
station_corrections.add_argument(
    "--plot",
    action="store_true",
    default=False,
    help="plot station correction results and save to rundir",
)
rundir_corrections = station_corrections.add_argument(
    "rundir",
    type=Path,
    help="path of existing run",
)

modules = subparsers.add_parser(
    "modules",
    help="list available modules",
    description="list all available modules",
)
modules.add_argument(
    "--json",
    "-j",
    type=str,
    help="print module's JSON config",
    default="",
)

serve = subparsers.add_parser(
    "serve",
    help="start webserver and serve results from an existing run",
    description="start a webserver and serve detections and results from a run",
)
serve.add_argument(
    "rundir",
    type=Path,
    help="rundir to serve",
)

subparsers.add_parser(
    "clear-cache",
    help="clear the cach directory",
    description="clear all data in the cache directory",
)

dump_schemas = subparsers.add_parser(
    "dump-schemas",
    help="dump data models to json-schema (development)",
    description="dump data models to json-schema, "
    "this is for development purposes only",
)
dump_dir = dump_schemas.add_argument(
    "folder",
    type=Path,
    help="folder to dump schemas to",
)


try:
    import argcomplete
    from argcomplete.completers import DirectoriesCompleter, FilesCompleter

    config_arg.completer = FilesCompleter(["*.json"])
    rundir_continue.completer = DirectoriesCompleter()
    rundir_features.completer = DirectoriesCompleter()
    rundir_corrections.completer = DirectoriesCompleter()
    dump_dir.completer = DirectoriesCompleter()

    argcomplete.autocomplete(parser)
except ImportError:
    pass


def main() -> None:
    from rich import box
    from rich.progress import track
    from rich.prompt import IntPrompt
    from rich.table import Table

    from qseek.console import console
    from qseek.search import Search
    from qseek.server import WebServer
    from qseek.utils import CACHE_DIR, load_insights, setup_rich_logging

    load_insights()
    args = parser.parse_args()

    setup_rich_logging(level=logging.INFO - args.verbose * 10)

    match args.command:
        case "config":
            config = Search()
            console.print_json(config.model_dump_json(by_alias=False, indent=2))

        case "search":
            search = Search.from_config(args.config)

            webserver = WebServer(search)

            async def run() -> None:
                http = asyncio.create_task(webserver.start())
                await search.start(force_rundir=args.force)
                await http

            asyncio.run(run())

        case "continue":
            search = Search.load_rundir(args.rundir)
            if search._progress.time_progress:
                console.rule(f"Continuing search from {search._progress.time_progress}")
            else:
                console.rule("Starting search from scratch")

            webserver = WebServer(search)

            async def run() -> None:
                http = asyncio.create_task(webserver.start())
                await search.start()
                await http

            asyncio.run(run())

        case "feature-extraction":
            search = Search.load_rundir(args.rundir)
            search.data_provider.prepare(search.stations)

            async def extract() -> None:
                iterator = asyncio.as_completed(
                    tuple(
                        search.add_magnitude_and_features(detection)
                        for detection in search._detections
                    )
                )
                for result in track(
                    iterator,
                    description="Extracting features",
                    total=search._detections.n_detections,
                ):
                    detection = await result
                    await detection.dump_detection(update=True)

                await search._detections.export_detections(
                    jitter_location=search.octree.smallest_node_size()
                )

            asyncio.run(extract())

        case "corrections":
            rundir = Path(args.rundir)
            from qseek.corrections.base import StationCorrections

            search = Search.load_rundir(rundir)

            corrections_modules = StationCorrections.get_subclasses()

            console.print("[bold]Available travel time corrections modules")
            for imodule, module in enumerate(corrections_modules):
                console.print(f"{imodule}: {module.__name__}")

            module_choice = IntPrompt.ask(
                "Choose station corrections module",
                choices=[str(i) for i in range(len(corrections_modules))],
                default="0",
                console=console,
            )
            corrections_class = corrections_modules[int(module_choice)]
            corrections = asyncio.run(corrections_class.prepare(rundir, console))
            search.corrections = corrections

            new_config_file = rundir.parent / f"{rundir.name}-corrections.json"
            console.print("writing new config file")
            console.print(
                "to use this config file, run [bold]`qseek search %s`",
                new_config_file,
            )
            new_config_file.write_text(search.model_dump_json(by_alias=False, indent=2))

        case "serve":
            search = Search.load_rundir(args.rundir)
            webserver = WebServer(search)

            loop = asyncio.get_event_loop()
            loop.create_task(webserver.start())
            loop.run_forever()

        case "clear-cache":
            logger.info("clearing cache directory %s", CACHE_DIR)
            shutil.rmtree(CACHE_DIR)

        case "modules":
            from qseek.corrections.base import StationCorrections
            from qseek.features.base import FeatureExtractor
            from qseek.magnitudes.base import EventMagnitudeCalculator
            from qseek.tracers.base import RayTracer
            from qseek.waveforms.base import WaveformProvider

            table = Table(box=box.SIMPLE, header_style=None)

            table.add_column("Module")
            table.add_column("Description")

            module_classes = (
                RayTracer,
                FeatureExtractor,
                EventMagnitudeCalculator,
                WaveformProvider,
                StationCorrections,
            )

            if args.json:
                for module in module_classes:
                    for subclass in module.get_subclasses():
                        if subclass.__name__ == args.json:
                            console.print_json(subclass().model_dump_json(indent=2))
                            parser.exit()
                else:
                    parser.error(f"unknown module: {args.json}")

            def is_insight(module: type) -> bool:
                return "insight" in module.__module__

            for modules in module_classes:
                table.add_row(f"[bold]{modules.__name__}")
                for module in modules.get_subclasses():
                    name = module.__name__
                    if is_insight(module):
                        name += " 🔑"
                    table.add_row(f" {name}", module.__doc__, style="dim")
                table.add_section()

            console.print(table)

        case "dump-schemas":
            import json

            from qseek.models.detection import EventDetections

            if not args.folder.exists():
                raise EnvironmentError(f"folder {args.folder} does not exist")

            file = args.folder / "search.schema.json"
            print(f"writing JSON schemas to {args.folder}")
            file.write_text(json.dumps(Search.model_json_schema(), indent=2))

            file = args.folder / "detections.schema.json"
            file.write_text(json.dumps(EventDetections.model_json_schema(), indent=2))
        case _:
            parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()