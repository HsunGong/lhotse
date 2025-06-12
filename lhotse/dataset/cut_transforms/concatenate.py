import random
from typing import Optional, Sequence

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import Seconds, fastcopy


class CutConcatenate:
    """
    A transform on batch of cuts (``CutSet``) that concatenates the cuts to minimize the total amount of padding;
    e.g. instead of creating a batch with 40 examples, we will merge some of the examples together
    adding some silence between them to avoid a large number of padding frames that waste the computation.
    """

    def __init__(
        self,
        gap: Seconds = 1.0,
        duration_factor: float = 1.0,
        max_duration: Optional[Seconds] = None,
        seed: Optional[int] = None,
        change_symbol: Optional[str] = None,
    ) -> None:
        """
        CutConcatenate's constructor.

        :param gap: The duration of silence in seconds that is inserted between the cuts;
            it's goal is to let the model "know" that there are separate utterances in a single example.
        :param duration_factor: Determines the maximum duration of the concatenated cuts;
            by default it's 1, setting the limit at the duration of the longest cut in the batch.
        :param max_duration: If a value is given (in seconds), the maximum duration of concatenated cuts
            is fixed to the value while duration_factor is ignored.
        :param seed: If given, the cuts will be shuffled using this seed before concatenation.
        :param change_symbol: If given, this symbol will be added to the text of the supervision between two cuts
        """
        self.gap = gap
        self.duration_factor = duration_factor
        self.max_duration = max_duration
        self.seed = seed
        self.change_symbol = change_symbol

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.seed is None:
            cuts = cuts.sort_by_duration(ascending=False)
        else:
            rng_state = random.Random(self.seed)
            cuts = cuts.shuffle(rng_state)

        return concat_cuts(
            cuts,
            gap=self.gap,
            max_duration=self.max_duration
            if self.max_duration
            else cuts[0].duration * self.duration_factor,
            change_symbol=self.change_symbol,
        )


def concat_cuts(
    cuts: Sequence[Cut],
    gap: Seconds = 1.0,
    max_duration: Optional[Seconds] = None,
    change_symbol: Optional[str] = None,
) -> CutSet:
    """
    We're going to concatenate the cuts to minimize the amount of total padding frames used.
    This means that some samples in the batch will be merged together into one sample,
    separated by an interval of silence.
    This is actually solving a knapsack problem.
    In this initial implementation we're using a greedy approach:
    going from the back (i.e. the shortest cuts) we'll try to concat them to the longest cut
    that still has some "space" at the end.

    :param cuts: a list of cuts to pack.
    :param gap: the duration of silence inserted between concatenated cuts.
    :param max_duration: the maximum duration for the concatenated cuts
        (by default set to the duration of the first cut).
    :param change_symbol: if given, this symbol will be added to the text of the supervision
    :return a list of packed cuts.
    """
    if len(cuts) <= 1:
        # Nothing to do.
        return CutSet.from_cuts(cuts)
    max_duration = cuts[0].duration if max_duration is None else max_duration
    current_idx = 0
    while True:
        can_fit = False
        shortest = cuts[-1]
        for idx in range(current_idx, len(cuts) - 1):
            cut = cuts[current_idx]
            _gap = random.uniform(0, gap) if change_symbol else gap
            can_fit = cut.duration + _gap + shortest.duration <= max_duration
            if can_fit:
                if change_symbol:
                    assert (
                        len(cut.supervisions) == 1
                    ), "CutConcatenate transform requires cuts with exactly one supervision per cut."
                    cut = fastcopy(
                        cut,
                        supervisions=[
                            fastcopy(
                                cut.supervisions[0],
                                text=cut.supervisions[0].text + change_symbol,
                            )
                        ],
                    )
                cuts[current_idx] = cut.pad(cut.duration + _gap).append(shortest)
                cuts = cuts[:-1]
                break
            current_idx += 1
        if not can_fit:
            break
    return CutSet.from_cuts(cuts)
