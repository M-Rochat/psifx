import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.diarization.pyannote.command import PyannoteCommand
from psifx.audio.diarization.tool import DiarizationTool


class DiarizationCommand(Command):
    """
    Tools for diarizing audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "pyannote", PyannoteCommand)
        register_command(subparsers, "visualization", VisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class VisualizationCommand(Command):
    """
    Tool for visualizing the diarization of a track.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the input diarization file, such as '/path/to/diarization.rttm'",
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the output visualization file, such as '/path/to/visualization.png'",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        tool = DiarizationTool(
            device="cpu",
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.visualization(
            diarization_path=args.diarization,
            visualization_path=args.visualization,
        )
        del tool
