import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import joblib
from tqdm.auto import tqdm

import lhotse
from lhotse import CutSet, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.cut.mono import MonoCut
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

from .reazonspeech import normalize


def parse_utterance(
    item: Any,
) -> Optional[Tuple[Recording, SupervisionSegment, MonoCut]]:
    """
    Process a single utterance from the ReazonSpeech dataset.
    :param item: The utterance to process.
    :return: A tuple containing the Recording and SupervisionSegment.
    """
    id_name = Path(item[0]).stem
    try:
        with lhotse.audio_backend("LibsndfileBackend"):
            recording = Recording.from_file(item[0], recording_id=id_name)

        segments = SupervisionSegment(
            id=id_name,
            recording_id=id_name,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Japanese",
            text=normalize(item[1]),
        )

        cut = MonoCut(
            id=id_name,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[segments],
        )
        return recording, segments, cut
    except Exception as e:
        logging.warning(f"Failed to process {item[0]}: {e}")
        return None


def prepare_reazonspeech_v2(
    dataset_parts: Union[str, Sequence[str]],
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int = 1,
):
    if isinstance(dataset_parts, str):
        part = dataset_parts
    else:
        for part in dataset_parts:
            prepare_reazonspeech_v2(part, corpus_dir, output_dir, num_jobs)
        return

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) / part
    output_dir.mkdir(parents=True, exist_ok=True)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    logging.info(f"Preparing ReazonSpeech {part} data to {output_dir}")

    with open(
        output_dir / f"recordings.jsonl", "w", encoding="utf-8"
    ) as rec_writer, open(
        output_dir / f"supervisions.jsonl", "w", encoding="utf-8"
    ) as sup_writer, open(
        output_dir / f"cuts.jsonl", "w", encoding="utf-8"
    ) as cut_writer, open(
        corpus_dir / f"tsv/{part}.tsv", "r", encoding="utf-8"
    ) as f:

        def wrapper():
            for item in csv.reader(f, delimiter="\t"):
                item[0] = (corpus_dir / item[0]).absolute().as_posix()
                yield item

        for ret in joblib.Parallel(
            n_jobs=num_jobs,
            verbose=10,
            backend="loky",
            batch_size=128,
            pre_dispatch=10 * num_jobs,
            return_as="generator",
        )(joblib.delayed(parse_utterance)(item) for item in wrapper()):
            if ret is None:
                continue
            recording, segment, cut = ret
            print(recording.to_dict(), file=rec_writer, flush=True)
            print(segment.to_dict(), file=sup_writer, flush=True)
            print(cut.to_dict(), file=cut_writer, flush=True)
    logging.info(f"ReazonSpeech {part} data preparation is done.")
