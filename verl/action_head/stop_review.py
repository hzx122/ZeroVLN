"""Bounded stop-decision review using only the instruction and existing observation.

This is a candidate decision policy, not a parser repair or an arrival oracle.
The caller retains the original call budget and accounts for a deferred stop
as one zero-motion tool attempt. A confirmed stop is never repeatedly challenged.
"""
from dataclasses import dataclass


@dataclass
class StopReview:
    used: bool = False

    def consider(self, *, instruction: str, remaining_calls: int,
                 remaining_decisions: int) -> dict | None:
        """Return one review request, or None to execute the proposed stop.

        remaining_* are budgets AFTER the current stop attempt. Never defer
        when the agent would have no opportunity to confirm within its budget.
        A fresh instance must be created for each episode.
        """
        if self.used or remaining_calls < 1 or remaining_decisions < 1:
            return None
        self.used = True
        return {
            'schema': 'stop_decision_review_v1',
            'tool': 'stop',
            'ok': False,
            'return_reason': 'stop_confirmation_requested',
            'steps_used': 0,
            'robot_moved': False,
            'stop_executed': False,
            'instruction': instruction,
            'review': (
                'Check this proposed stop once against the unchanged current view '
                'and original instruction. Distinguish the final destination '
                '(including its identifying relations) from a local approach waypoint '
                'or source object. Check whether the view supports being beside that '
                'destination rather than only facing or seeing it across the room. '
                'Tool execution, travel distance, and collision alone do not establish '
                'arrival. If the destination and proximity are supported, call stop '
                'again with the specific visual evidence. If not, choose the next '
                'local approach or view adjustment. You are not required to move '
                'merely to satisfy this check. Reply with exactly one concise tool call.'
            ),
        }
