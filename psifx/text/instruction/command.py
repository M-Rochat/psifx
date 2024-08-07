import argparse
from psifx.text.instruction.tool import InstructionTool
from psifx.text.llm.tool import LLMTool
from psifx.utils.command import Command
from psifx.text.llm.command import AddLLMArgument
from pathlib import Path


class InstructionCommand(Command):
    """
    Command-line interface for custom instructions
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
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
        parser.add_argument(
            '--input',
            type=str,
            required=True,
            help="path to the input .txt or .csv file")
        parser.add_argument(
            '--output',
            type=str,
            required=True,
            help="path to the output .txt or .csv file")
        parser.add_argument(
            '--instruction',
            type=str,
            required=True,
            help="path to a .yaml file containing the prompt and parser")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        llm = LLMTool().llm_from_yaml(args.llm)
        chains = LLMTool().chains_from_yaml(llm, args.instruction)

        tool = InstructionTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
            chain=next(iter(chains.values()))
        )

        path = Path(args.input)
        if path.suffix == '.txt':
            tool.apply_to_txt(input_path=args.input,
                              output_path=args.output)
        elif path.suffix == '.csv':
            tool.apply_to_csv(input_path=args.input,
                              output_path=args.output)
        else:
            raise NameError(f"Input path should be .txt or .csv, got {args.input} instead.")
