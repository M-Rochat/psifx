"""large language model command-line interface."""

import argparse
from pathlib import Path
from psifx.io.yaml import YAMLReader

def format_llm_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """
    Format an Argument Namespace for LLM use.
    :param args:  Argument Namespace.
    :return: A formatted Argument Namespace.
    """
    llm_kw_args = YAMLReader.read(args.model_config) if args.model_config else {}
    assert 'api_key' not in llm_kw_args, f"api_key should not be specified in the model config file, please provide it with --api_key or as an environment variable"
    for key in ('provider', 'model'):
        assert not (key in llm_kw_args and key in args), f"{key} is specified both as an argument and in the model config file, please provide only one"
    if 'provider' not in llm_kw_args:
        llm_kw_args['provider'] = args.provider if 'provider' in args else 'ollama'
    if 'model' not in llm_kw_args:
        llm_kw_args['model'] = args.model if 'model' in args else 'llama3.1:8b'
    if args.api_key is not None:
        llm_kw_args['api_key'] = args.api_key
    return argparse.Namespace(**llm_kw_args)


def add_llm_argument(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for large language model.
    :param parser: An argument parser.
    """
    parser.add_argument(
        '--provider',
        type=str,
        default=argparse.SUPPRESS,
        choices=['ollama', 'hf', 'openai', 'anthropic'],
        required=False,
        help="The large language model provider. Choices are 'ollama', 'hf', 'openai', or 'anthropic'. Default is 'ollama'."
    )
    parser.add_argument(
        '--model',
        type=str,
        default=argparse.SUPPRESS,
        required=False,
        help="The large language model to use. This depends on the provider."
    )
    parser.add_argument(
        '--model_config',
        type=Path,
        required=False,
        help="Path to the model .yaml configuration file."
    )
    parser.add_argument(
        '--api_key',
        type=str,
        required=False,
        help="Corresponding API key for 'hf', 'openai', or 'anthropic'."
    )
