"""
Task spawner stub.

Generates spatial tasks with location, deadline, and completion status.
Phase 1 only — auction wiring is deferred to Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    """A spatial task requiring agent visitation."""

    task_id: int
    location: np.ndarray
    deadline: float
    completed: bool = False
    assigned_to: int | None = None


class TaskSpawner:
    """
    Generates tasks uniformly within the spatial domain.

    Parameters
    ----------
    grid_width : float
        Domain width.
    grid_height : float
        Domain height.
    """

    def __init__(self, grid_width: float, grid_height: float) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self._next_id: int = 0

    def spawn(
        self,
        count: int,
        deadline: float,
        rng: np.random.Generator,
    ) -> list[Task]:
        """Generate *count* random tasks with the given deadline."""
        tasks: list[Task] = []
        for _ in range(count):
            loc = np.array([
                rng.uniform(0.0, self.grid_width),
                rng.uniform(0.0, self.grid_height),
            ])
            tasks.append(Task(task_id=self._next_id, location=loc, deadline=deadline))
            self._next_id += 1
        return tasks
