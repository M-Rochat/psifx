from typing import Sequence, Union

from pathlib import Path

import pandas as pd
from pandas import DataFrame


class RTTMReader(object):
    @staticmethod
    def read(
        path: Union[str, Path],
    ) -> DataFrame:
        path = Path(path)

        assert path.suffix == ".rttm"

        dataframe = pd.read_csv(
            path,
            sep=" ",
            header=None,
            names=[
                "type",
                "file_stem",
                "channel",
                "start",
                "duration",
                "orthography",
                "speaker_type",
                "speaker_name",
                "confidence_score",
                "signal_lookahead_time",
            ],
        )
        dataframe["end"] = dataframe["start"] + dataframe["duration"]

        return dataframe


class RTTMWriter(object):
    @staticmethod
    def write(
        path: Union[str, Path],
        type: Sequence[str],
        file_stem: Sequence[str],
        channel: Sequence[int],
        start: Sequence[float],
        duration: Sequence[float],
        orthography: Sequence[str],
        speaker_type: Sequence[str],
        speaker_name: Sequence[str],
        confidence_score: Sequence[Union[float, str]],
        signal_lookahead_time: Sequence[Union[float, str]],
        overwrite: bool = False,
    ):
        path = Path(path)

        assert path.suffix == ".rttm"
        assert (
            len(type)
            == len(file_stem)
            == len(channel)
            == len(start)
            == len(duration)
            == len(orthography)
            == len(speaker_type)
            == len(speaker_name)
            == len(confidence_score)
            == len(signal_lookahead_time)
        )

        dataframe = pd.DataFrame.from_dict(
            {
                "type": type,
                "file_stem": file_stem,
                "channel": channel,
                "start": start,
                "duration": duration,
                "orthography": orthography,
                "speaker_type": speaker_type,
                "speaker_name": speaker_name,
                "confidence_score": confidence_score,
                "signal_lookahead_time": signal_lookahead_time,
            }
        )

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, sep=" ", header=False, float_format="%.3f")