"""
The Emilia dataset is constructed from a vast collection of speech data sourced
from diverse video platforms and podcasts on the Internet, covering various
content genres such as talk shows, interviews, debates, sports commentary, and
audiobooks. This variety ensures the dataset captures a wide array of real
human speaking styles. The initial version of the Emilia dataset includes a
total of 101,654 hours of multilingual speech data in six different languages:
English, French, German, Chinese, Japanese, and Korean.

See also
https://emilia-dataset.github.io/Emilia-Demo-Page/

Please note that Emilia does not own the copyright to the audio files; the
copyright remains with the original owners of the videos or audio. Users are
permitted to use this dataset only for non-commercial purposes under the
CC BY-NC-4.0 license.

Please refer to
https://huggingface.co/datasets/amphion/Emilia-Dataset
or
https://openxlab.org.cn/datasets/Amphion/Emilia
to download the dataset.

Note that you need to apply for downloading.

"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import joblib
from tqdm.auto import tqdm

from lhotse import CutSet, MonoCut
from lhotse.audio import Recording
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Pathlike


def _parse_utterance(
    data_dir: Path,
    json_file: str,
) -> Optional[Tuple[Recording, SupervisionSegment, MonoCut]]:
    """
    :param data_dir: Path to the data directory
    :param line: dict, it looks like below::

        {
          "id": "DE_B00000_S00000_W000029",
          "wav": "DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000029.mp3",
          "text": " Und es gibt auch einen Stadtplan von Tegun zu sehen.",
          "duration": 3.228,
          "speaker": "DE_B00000_S00000",
          "language": "de",
          "dnsmos": 3.3697
        }

    :return: a tuple of "recording" and "supervision"
    """
    meta_data = json.load(open(json_file))
    if "id" in meta_data:
        unique_id = meta_data["id"]
        lang, session, spk, idx = unique_id.split("_")
    else:  # emilia-yodas
        unique_id = meta_data["_id"]
        _splits = unique_id.split("_")
        lang = _splits[0]
        idx = _splits[-1]
        session = "_".join(_splits[1:-1])

    recording = Recording.from_file(
        path=Path(json_file).with_suffix(".mp3"),
        recording_id=unique_id,
    )

    segment = SupervisionSegment(
        id=unique_id,
        recording_id=unique_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        text=meta_data["text"],
        language=meta_data["language"],
        speaker=meta_data["speaker"],
        custom={"dnsmos": meta_data["dnsmos"]},
    )

    cut = MonoCut(
        id=recording.id,
        recording=recording,
        start=0,
        duration=recording.duration,
        supervisions=[segment],
        channel=0,
        custom={"session": session, "idx": idx},
    )

    return recording, segment, cut


def prepare_emilia(
    corpus_dir: Pathlike,
    lang: str,
    num_jobs: int,
    output_dir: Optional[Pathlike] = None,
) -> None:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
                       We assume the directory has the following structure:
                       corpus_dir/raw/openemilia_all.tar.gz,
                       corpus_dir/raw/DE,
                       corpus_dir/raw/DE/DE_B00000.jsonl,
                       corpus_dir/raw/DE/DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000000.mp3,
                       corpus_dir/raw/EN, etc.
    :param lang: str, one of en, zh, de, ko, ja, fr
    :param num_jobs: int, number of threads for processing jsonl files
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: The CutSet containing the data for the given language.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if lang is None:
        raise ValueError("Please provide --lang")
    lang_uppercase = lang.upper()
    if lang_uppercase not in ("DE", "EN", "FR", "JA", "KO", "ZH", "ALL"):
        raise ValueError(
            "Please provide a valid language. "
            f"Choose from de, en, fr, ja, ko, zh, all. Given: {lang}"
        )

    if lang_uppercase == "ALL":
        for lang in ("DE", "EN", "FR", "JA", "KO", "ZH"):
            prepare_emilia(
                corpus_dir=corpus_dir,
                lang=lang,
                num_jobs=num_jobs,
                output_dir=output_dir,
            )
        return

    lang_dir = corpus_dir / lang_uppercase
    assert output_dir is not None, "Please provide --output-dir"
    output_dir = Path(output_dir) / lang_uppercase
    output_dir.mkdir(parents=True, exist_ok=True)

    with subprocess.Popen(
        ["find", str(lang_dir), "-maxdepth", "3", "-name", "*.json", "-print"],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc, open(output_dir / f"recordings.jsonl", "w") as frecordings, open(
        output_dir / f"supervisions.jsonl", "w"
    ) as fsupervisions, open(
        output_dir / f"cuts.jsonl", "w"
    ) as fcuts:
        pbar = tqdm(desc=f"Processing in {lang_dir}")
        for ret in joblib.Parallel(n_jobs=num_jobs, return_as="generator")(
            joblib.delayed(_parse_utterance)(
                data_dir=lang_dir,
                json_file=json_file.strip(),
            )
            for json_file in proc.stdout
        ):
            if ret is None:
                continue

            recording, supervision, cut = ret
            frecordings.write(
                json.dumps(recording.to_dict(), ensure_ascii=False) + "\n"
            )
            fsupervisions.write(
                json.dumps(supervision.to_dict(), ensure_ascii=False) + "\n"
            )
            fcuts.write(json.dumps(cut.to_dict(), ensure_ascii=False) + "\n")
            pbar.update(1)
