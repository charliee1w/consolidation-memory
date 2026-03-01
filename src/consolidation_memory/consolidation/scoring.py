"""Surprise score adjustment and adaptive scoring.

Handles per-episode surprise score updates: access-count boosting,
inactivity decay, and clamping to configured min/max bounds.
"""

import logging
from datetime import datetime, timezone

from consolidation_memory.config import get_config
from consolidation_memory.database import (
    get_active_episodes_paginated,
    get_median_access_count,
    update_surprise_scores,
)

logger = logging.getLogger(__name__)


def _adjust_surprise_scores() -> int:
    cfg = get_config()
    # Get median access count via single SQL query instead of loading all episodes
    median_access = get_median_access_count()

    now = datetime.now(timezone.utc)
    total_updates = 0
    total_processed = 0
    page_size = 1000
    offset = 0

    while True:
        episodes = get_active_episodes_paginated(offset=offset, limit=page_size)
        if not episodes:
            break

        total_processed += len(episodes)
        updates = []

        for ep in episodes:
            original = ep["surprise_score"]
            new_score = original
            access = ep["access_count"]

            if access > median_access and median_access > 0:
                excess = access - median_access
                # Absolute target rather than additive boost — prevents
                # cumulative inflation across repeated consolidation runs.
                target = original + min(excess * cfg.SURPRISE_BOOST_PER_ACCESS, 0.15)
                new_score = max(new_score, target)

            try:
                last_update = datetime.fromisoformat(ep["updated_at"])
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=timezone.utc)
                days_inactive = (now - last_update).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_inactive = 0

            # Only decay episodes that have been consolidated (their knowledge is
            # captured in a topic document).  Unconsolidated episodes are the sole
            # record of their information and should never be decayed into
            # obscurity.  This breaks the positive feedback loop where low-access
            # episodes decay → rank lower → get accessed even less → decay more.
            is_consolidated = ep.get("consolidated", 0) == 1
            if access == 0 and days_inactive >= cfg.SURPRISE_DECAY_INACTIVE_DAYS and is_consolidated:
                new_score -= cfg.SURPRISE_DECAY_RATE

            new_score = max(cfg.SURPRISE_MIN, min(cfg.SURPRISE_MAX, new_score))

            if abs(new_score - original) >= 0.005:
                updates.append((round(new_score, 4), ep["id"]))

        if updates:
            update_surprise_scores(updates)
            total_updates += len(updates)

        if len(episodes) < page_size:
            break
        offset += page_size

    if total_updates:
        logger.info(
            "Adjusted surprise scores for %d/%d episodes (median_access=%.1f)",
            total_updates,
            total_processed,
            median_access,
        )

    return total_updates
