"""HuggingFace Whisper transcription command-line interface."""
import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.huggingface.tool import HuggingFaceWhisperTool


class HuggingFaceWhisperCommand(Command):
    """
    Command-line interface for running HuggingFace Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", HuggingFaceWhisperInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class HuggingFaceWhisperInferenceCommand(Command):
    """
    Command-line interface for transcribing an audio track with HuggingFace Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--audio",
            type=Path,
            required=True,
            help="path to the input audio file, such as ``/path/to/audio.wav``",
        )
        parser.add_argument(
            "--transcription",
            type=Path,
            required=True,
            help="path to the output transcription file, such as ``/path/to/transcription.vtt``",
        )
        parser.add_argument(
            "--language",
            type=str,
            default=None,
            help="language of the audio, if ignore, the model will try to guess it, "
                 "it is advised to specify it",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="openai/whisper-small",
            help="name of the model, check https://huggingface.co/models?other=whisper",
        )
        parser.add_argument(
            "--api_token",
            type=str,
            default=None,
            help="API token for downloading the model from HuggingFace",
        )
        parser.add_argument(
            "--translate_to_english",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="whether to transcribe the audio in its original language or"
                 " to translate it to english",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
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
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        tool = HuggingFaceWhisperTool(
            model_name=args.model_name,
            api_token=args.api_token,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        tool.inference(
            audio_path=args.audio,
            transcription_path=args.transcription,
            task="transcribe" if not args.translate_to_english else "translate",
            language=args.language,
        )
        del tool
