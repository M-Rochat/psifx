"""audio manipulation tool."""

from typing import Sequence, Union

from pathlib import Path
import ffmpeg

from pydub import AudioSegment

from psifx.tool import Tool


class ManipulationTool(Tool):
    """
    audio manipulation tool.

    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

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

    def extraction(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
    ):
        """
        Extracts the audio track from a video.

        :param video_path: Path to the video.
        :param audio_path: Path to the audio track.
        :return:
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"audio   =   {audio_path}")

        if audio_path.exists():
            if self.overwrite:
                audio_path.unlink()
            else:
                raise FileExistsError(audio_path)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg.input(str(video_path))
            .audio.output(str(audio_path), **{"q:a": 0, "ar": 32000})
            .overwrite_output()
            .run(quiet=self.verbose <= 1)
        )

    def convert(
        self,
        audio_path: Union[str, Path],
        mono_audio_path: Union[str, Path],
    ):
        """
        Converts an audio track to a .wav audio track with 32kHz sample rate.

        :param audio_path: Path to the audio track.
        :param mono_audio_path: Path to the converted audio track.
        :return:
        """
        audio_path = Path(audio_path)
        mono_audio_path = Path(mono_audio_path)

        assert mono_audio_path.suffix == ".wav"

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"mono_audio      =   {mono_audio_path}")

        mono_audios = [
            audio.apply_gain(-audio.max_dBFS - 6.0)
            for audio in AudioSegment.from_file(audio_path).split_to_mono()
        ]
        mono_audio = mono_audios[0]
        for audio in mono_audios:
            mono_audio.overlay(audio)

        if mono_audio_path.exists():
            if self.overwrite:
                mono_audio_path.unlink()
            else:
                raise FileExistsError(mono_audio_path)
        mono_audio_path.parent.mkdir(parents=True, exist_ok=True)

        mono_audio.export(mono_audio_path, format="wav")

    def split(
            self,
            stereo_audio_path: Union[str, Path],
            left_audio_path: Union[str, Path],
            right_audio_path: Union[str, Path],
    ):
        """
        Splits a stereo audio track into two mono audio tracks.

        :param stereo_audio_path: Path to the stereo audio track.
        :param left_audio_path: Path to the left channel mono audio track.
        :param right_audio_path: Path to the right channel mono audio track.
        :return:
        """
        stereo_audio_path = Path(stereo_audio_path)
        left_audio_path = Path(left_audio_path)
        right_audio_path = Path(right_audio_path)

        if self.verbose:
            print(f"stereo_audio   =   {stereo_audio_path}")
            print(f"left_audio     =   {left_audio_path}")
            print(f"right_audio    =   {right_audio_path}")

        stereo_audio = AudioSegment.from_file(stereo_audio_path)
        assert stereo_audio.channels == 2, "Input audio is not stereo."

        left_channel, right_channel = stereo_audio.split_to_mono()

        if left_audio_path.exists():
            if self.overwrite:
                left_audio_path.unlink()
            else:
                raise FileExistsError(left_audio_path)
        if right_audio_path.exists():
            if self.overwrite:
                right_audio_path.unlink()
            else:
                raise FileExistsError(right_audio_path)

        left_audio_path.parent.mkdir(parents=True, exist_ok=True)
        right_audio_path.parent.mkdir(parents=True, exist_ok=True)

        left_channel.export(left_audio_path, format="wav")
        right_channel.export(right_audio_path, format="wav")

    def mixdown(
        self,
        mono_audio_paths: Sequence[Union[str, Path]],
        mixed_audio_path: Union[str, Path],
    ):
        """
        Mixes multiple mono audio tracks.

        :param mono_audio_paths: Paths to mono audio tracks.
        :param mixed_audio_path: Path to the mixed audio track.
        :return:
        """
        mono_audio_paths = [Path(path) for path in mono_audio_paths]
        mixed_audio_path = Path(mixed_audio_path)

        if self.verbose:
            print(f"mono_audios     =   {[str(path) for path in mono_audio_paths]}")
            print(f"mixed_audio     =   {mixed_audio_path}")

        mono_audios = [AudioSegment.from_wav(path) for path in mono_audio_paths]
        for audio in mono_audios:
            assert audio.channels == 1, f"Audio file {audio} is not mono"
        mono_audios = [audio.apply_gain(-audio.max_dBFS - 6.0) for audio in mono_audios]
        mixed_audio = mono_audios[0]
        for audio in mono_audios[1:]:
            mixed_audio.overlay(audio)

        if mixed_audio_path.exists():
            if self.overwrite:
                mixed_audio_path.unlink()
            else:
                raise FileExistsError(mixed_audio_path)
        mixed_audio_path.parent.mkdir(parents=True, exist_ok=True)

        mixed_audio.export(mixed_audio_path, format="wav")

    def normalization(
        self,
        audio_path: Union[str, Path],
        normalized_audio_path: Union[str, Path],
    ):
        """
        Normalizes an audio track.

        :param audio_path: Path to the audio track.
        :param normalized_audio_path: Path to the normalized audio track.
        :return:
        """
        audio_path = Path(audio_path)
        normalized_audio_path = Path(normalized_audio_path)

        if self.verbose:
            print(f"audio               =   {audio_path}")
            print(f"normalized_audio    =   {normalized_audio_path}")

        audio = AudioSegment.from_wav(audio_path)
        normalized_audio = audio.apply_gain(-audio.max_dBFS)

        if normalized_audio_path.exists():
            if self.overwrite:
                normalized_audio_path.unlink()
            else:
                raise FileExistsError(normalized_audio_path)
        normalized_audio_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_audio.export(normalized_audio_path, format="wav")
