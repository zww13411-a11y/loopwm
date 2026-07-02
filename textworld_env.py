"""
Synthetic Text Environment — ScienceWorld-like but zero external deps.

Generates structured text observations and discrete actions with
controllable dynamics, so we can verify LoopWM training convergence
without Java/ScienceWorld.

Each "task" is a simple state machine:
  state → action → next_state (with transition matrix)
  observation = template(state)  (e.g., "You are in {room}. You see {items}.")
  action = one of N discrete actions

We know the underlying dynamics exactly → can measure exact
reconstruction accuracy, not just proxy metrics.
"""

from __future__ import annotations

import random
from typing import Any


# ──────────────────────────────────────────────
#  Room-and-Items: a small text-based world
# ──────────────────────────────────────────────

ROOMS = ["kitchen", "workshop", "greenhouse", "bedroom"]
ITEMS = ["red solid", "blue liquid", "green plant", "metal tool", "glass beaker"]
ACTIONS = [
    "look around",
    "pick up red solid",
    "pick up blue liquid",
    "heat red solid",
    "heat blue liquid",
    "mix beaker contents",
    "move to kitchen",
    "move to workshop",
    "move to greenhouse",
    "move to bedroom",
]
ACTION_LOOK = 0
ACTION_MOVE_KITCHEN = 6
ACTION_MOVE_WORKSHOP = 7
ACTION_MOVE_GREENHOUSE = 8
ACTION_MOVE_BEDROOM = 9


VALID_ACTIONS_PER_PHASE = {
    "default": list(range(10)),
}


class TextWorld:
    """
    A room-and-items text environment.

    State = (room: int, items: dict[item_name: location])

    Observations are structured text templates.
    Actions change location or interact with items.

    Has multiple "task types" with different difficulty:
      - `navigate`: reach a target room (simple, short horizon)
      - `collect`: collect specific items (medium)
      - `transform`: heat/mix items to change their state (complex)
    """

    def __init__(self, task_type: str = "navigate", seed: int = 42):
        self.rng = random.Random(seed)
        self.task_type = task_type
        self.max_steps = 100
        self._step = 0
        self._done = False
        self._reward = 0.0

        # State
        self.room = 0  # index into ROOMS
        self.items: dict[str, str] = {}  # item_name -> "raw" | "heated" | "mixed"
        self.inventory: list[str] = []

        # Task-specific targets
        self._target_room: int = 0
        self._target_items: list[str] = []
        self._setup_task()

    def _setup_task(self):
        if self.task_type == "navigate":
            # Start in one room, need to reach another
            self.room = self.rng.randint(0, len(ROOMS) - 1)
            possible_targets = [i for i in range(len(ROOMS)) if i != self.room]
            self._target_room = self.rng.choice(possible_targets)

        elif self.task_type == "collect":
            # Need to pick up specific items
            self.room = 0
            self._target_items = self.rng.sample(ITEMS, 2)
            for item in ITEMS:
                self.items[item] = "raw"

        elif self.task_type == "transform":
            # Need to heat specific items
            self.room = 1  # workshop has heating equipment
            self._target_items = self.rng.sample(ITEMS[:3], 1)
            for item in ITEMS:
                self.items[item] = "raw"

    def reset(self):
        self._step = 0
        self._done = False
        self._reward = 0.0
        self._setup_task()
        return self._get_obs(), {}

    def _get_obs(self) -> str:
        lines = [f"You are in the {ROOMS[self.room]}."]

        # Items in this room
        room_items = ITEMS[:3] if self.room < 3 else ITEMS[2:]
        if room_items:
            items_str = ", ".join(room_items)
            lines.append(f"On the table you see: {items_str}.")

        # Inventory
        if self.inventory:
            lines.append(f"You are carrying: {', '.join(self.inventory)}.")

        # Task hint
        if self.task_type == "navigate":
            lines.append(f"Your goal is to reach the {ROOMS[self._target_room]}.")
        elif self.task_type == "collect":
            lines.append(f"You need to collect: {', '.join(self._target_items)}.")
        elif self.task_type == "transform":
            lines.append(f"You need to heat: {', '.join(self._target_items)}.")

        return "\n".join(lines)

    def _get_available_actions(self) -> list[int]:
        return list(range(len(ACTIONS)))

    def step(self, action_idx: int) -> tuple[str, float, bool, dict]:
        self._step += 1
        reward = 0.0

        action = ACTIONS[action_idx]

        if action == "look around":
            pass  # no state change

        elif action.startswith("move to"):
            target_room_name = action.split("move to ")[1]
            for i, r in enumerate(ROOMS):
                if r == target_room_name:
                    self.room = i
                    break

        elif action.startswith("pick up"):
            item = action.split("pick up ")[1]
            if item in self.items and item not in self.inventory:
                self.inventory.append(item)
                if self.task_type == "collect" and item in self._target_items:
                    reward += 1.0

        elif action.startswith("heat"):
            item = action.split("heat ")[1]
            if item in self.inventory:
                self.items[item] = "heated"
                if self.task_type == "transform" and item in self._target_items:
                    reward += 2.0

        elif action == "mix beaker contents":
            if len(self.inventory) >= 2:
                for item in self.inventory:
                    self.items[item] = "mixed"
                reward += 1.0

        # Check termination
        done = False
        extra_reward = 0.0
        if self.task_type == "navigate" and self.room == self._target_room:
            done = True
            extra_reward = 10.0
        elif self.task_type == "collect":
            if all(t in self.inventory for t in self._target_items):
                done = True
                extra_reward = 10.0
        elif self.task_type == "transform":
            if all(self.items.get(t) == "heated" for t in self._target_items):
                done = True
                extra_reward = 10.0

        if self._step >= self.max_steps:
            done = True

        self._done = done
        self._reward += reward + extra_reward

        return self._get_obs(), reward + extra_reward, done, {}


# ──────────────────────────────────────────────
#  Text tokenizer helper (shared encoder paths)
# ──────────────────────────────────────────────

class SimpleTokenizer:
    """
    Build a token → id vocabulary from all observed texts.

    tokenizes by splitting on whitespace and punctuation.
    """

    def __init__(self):
        # Reserve token 0 for padding
        self.token_to_id: dict[str, int] = {"<pad>": 0}
        self.id_to_token: dict[int, str] = {0: "<pad>"}
        self.vocab_size = 1
        self._frozen = False

    def _add_token(self, token: str):
        if token not in self.token_to_id and not self._frozen:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.vocab_size = len(self.token_to_id)

    def fit(self, texts: list[str]):
        for text in texts:
            tokens = text.replace(",", " ,").replace(".", " .").replace(":", " :").split()
            for t in tokens:
                self._add_token(t.lower())

    def freeze(self):
        self._frozen = True

    def encode(self, text: str) -> list[int]:
        tokens = text.replace(",", " ,").replace(".", " .").replace(":", " :").split()
        unk_id = self.token_to_id.get("<unk>", self.token_to_id.get("<pad>", 0))
        return [self.token_to_id.get(t.lower(), unk_id) for t in tokens]

    def vocab_size(self) -> int:
        return len(self.token_to_id)


# ──────────────────────────────────────────────
#  Dataset collector
# ──────────────────────────────────────────────

def collect_trajectories(
    num_episodes: int = 100,
    task_type: str = "navigate",
    max_steps: int = 30,
    seed: int = 42,
) -> dict:
    """
    Collect trajectories from TextWorld using random actions.

    Returns: {
        "observations": [str],
        "actions": [int],
        "rewards": [float],
        "dones": [bool],
        "episode_ids": [int],
    }
    """
    data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
    }

    for ep in range(num_episodes):
        env = TextWorld(task_type=task_type, seed=seed + ep)
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action = env.rng.randint(0, len(ACTIONS) - 1)
            next_obs, reward, done, _ = env.step(action)

            data["observations"].append(obs)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["dones"].append(int(done))
            data["episode_ids"].append(ep)

            obs = next_obs
            step += 1

    return data


def build_vocab(data: dict) -> SimpleTokenizer:
    """Build tokenizer from all observations and action texts."""
    tokenizer = SimpleTokenizer()
    tokenizer.fit(data["observations"] + [a.lower() for a in ACTIONS])
    tokenizer.freeze()
    return tokenizer
