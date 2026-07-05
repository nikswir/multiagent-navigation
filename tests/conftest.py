"""Shared test fixtures and the stage-1 / stage-2 split.

Stage 2 (heavy: downloads, GPU, long-running) is skipped unless RUN_STAGE2=1,
so the default `pytest` run stays fast, offline and deterministic.
"""

import os

import pytest

########################################
#          Stage-2 test gate           #
########################################

# Stage-2 tests are heavy (download data/weights, GPU, slow); skip them unless
# explicitly enabled (RUN_STAGE2=1) so the default suite stays offline and fast.
# Mark a heavy test with `@stage2` (import it from this module).
RUN_STAGE2 = os.environ.get("RUN_STAGE2") == "1"

stage2 = pytest.mark.skipif(
    not RUN_STAGE2,
    reason="set RUN_STAGE2=1 to run heavy stage-2 tests",
)
