from typing import Union

import shlex
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import time

import pandas as pd
import numpy as np

from psifx.video.face.inference_tool import BaseFaceAnalysisTool
from psifx.video.face.openface.fields import CLEAN_FIELDS, DIRTY_FIELDS
from psifx.utils import tar, timestamp


EXECUTABLE_PATH = "$(which FeatureExtraction)"
DEFAULT_OPTIONS = "-2Dfp -3Dfp -pdmparams -pose -aus -gaze -au_static"


class OpenFaceAnalysisTool(BaseFaceAnalysisTool):
    def __init__(
        self,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device="cpu",
            overwrite=overwrite,
            verbose=verbose,
        )

    def __call__(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(features_path, Path):
            features_path = Path(features_path)

        assert video_path.is_file()

        tmp_dir = Path(f"/tmp/TEMP_{time.time()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        args = f"{EXECUTABLE_PATH} -f {video_path} -out_dir {tmp_dir} {DEFAULT_OPTIONS}"

        if self.verbose:
            print("OpenFace will run with the following command:")
            print(f"{args}")
            print("It might take a while, depending on the length of the video and the")
            print("number of CPUs.")

        try:
            start = time.time()
            # WEIRD BUT PRINTS IF NOT CAPTURED
            # process = subprocess.run(
            #     args=shlex.split(args),
            #     check=True,
            #     capture_output=not self.verbose > 1,
            #     text=True,
            # )
            # MORE CORRECT BUT LESS INTERACTIVE
            process = subprocess.run(
                args=shlex.split(args),
                check=True,
                stdout=subprocess.PIPE if self.verbose > 1 else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if self.verbose > 1 else subprocess.DEVNULL,
                text=True,
            )
            end = time.time()

            if self.verbose > 1:
                print(process.stdout)
            if self.verbose:
                print(f"OpenFace took {timestamp.format_timestamp(end - start)}.")

        except subprocess.CalledProcessError as error:
            print(error.stdout)

        dirty_dataframe = pd.read_csv(tmp_dir / (video_path.stem + ".csv"))

        clean_dataframe = pd.DataFrame()
        clean_dataframe["index"] = dirty_dataframe["frame"] - 1
        for clean, dirty in zip(CLEAN_FIELDS, DIRTY_FIELDS):
            clean_dataframe[clean] = dirty_dataframe[dirty].values.tolist()

        n_rows, n_cols = clean_dataframe.shape
        features = {}
        for i in tqdm(
            range(n_rows),
            disable=not self.verbose,
        ):
            index = clean_dataframe["index"][i]
            features[f"{index: 015d}"] = {
                clean: np.array(clean_dataframe[clean][i]).flatten().tolist()
                for clean in CLEAN_FIELDS
            }

        shutil.rmtree(tmp_dir)

        if features_path.exists():
            if self.overwrite:
                features_path.unlink()
            else:
                raise FileExistsError(features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        tar.dump(
            dictionary={f"{k}.json": json.dumps(v) for k, v in features.items()},
            path=features_path,
        )


def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files, otherwise raises an error.",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verbosity of the script.",
    )
    args = parser.parse_args()

    tool = OpenFaceAnalysisTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    if args.video.is_file():
        video_path = args.video
        features_path = args.features
        tool(
            video_path=video_path,
            features_path=features_path,
        )
    elif args.video.is_dir():
        video_dir = args.video
        features_dir = args.features
        for video_path in sorted(video_dir.glob("*.mp4")):
            features_name = video_path.stem + ".tar.gz"
            features_path = features_dir / features_name
            tool(
                video_path=video_path,
                features_path=features_path,
            )
    else:
        raise ValueError("args.video is neither a file or a directory.")

    del tool


if __name__ == "__main__":
    cli_main()
