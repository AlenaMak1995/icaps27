#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Two interacting four-role fleet operators on one shared physical road.

#This module composes two local moving-bike worlds into one centralized
#product MDP.  Each fleet contains four controlled road users:
#Fleet 1: lead car, merging car, emergency vehicle, cyclist
#Fleet 2: lead car, merging car, emergency vehicle, cyclist

# the two fleets share the same road cells and lanes.  Cross-fleet vehicle collisions, cross-fleet close
#passes, and cross-fleet cyclist collisions are therefore represented in the
#joint transition model.

#The joint MDP has one occupation measure.  It retains eight role-cost rows and
#also exposes two operator-cost rows formed from user-supplied internal role
#weights.  A shared four-coordinate norm
#z = (tau, sigma_4ft, sigma_3ft, sigma_2ft)
#is applied separately to both operators through eight relaxable rows and a
#shared slack map R
#This is a centralized product-MDP model
#The action and state products grow rapidly; the experiment runner freezes the
#validated horizon, geometry, and internal role weights.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import importlib.util
import math
import sys
import time

import numpy as np
import scipy.sparse as sp


LocalAction = Any
TwoFleetAction = Tuple[Optional[LocalAction], Optional[LocalAction]]

def _load_module_from_path(module_name: str, path: Path) -> Any:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_domain2_moving_bike_model(
    *,
    moving_bike_file: Path,
    module_prefix: str = "rq2_domain2",
) -> Any:

    model = _load_module_from_path(
        f"{module_prefix}_moving_bike", Path(moving_bike_file)
    )
    if not hasattr(model, "make_e4_moving_bike_env"):
        raise AttributeError(
            "moving-bike model does not expose make_e4_moving_bike_env"
        )
    return model


# Joint state and world
@dataclass(frozen=True, order=True)
class TwoFleetState:
    """Joint state of the two four-role fleets plus cross-fleet monitors."""

    fleet1: Any
    fleet2: Any

    # Cross-fleet vehicle--vehicle collision monitor.
    cross_vehicle_collision_bad: bool = False

    # Cross-fleet events affecting Bike_1.
    bike1_curb_band_bad: bool = False
    bike1_close_pass_bad: bool = False
    bike1_collision_bad: bool = False

    # Cross-fleet events affecting Bike_2.
    bike2_curb_band_bad: bool = False
    bike2_close_pass_bad: bool = False
    bike2_collision_bad: bool = False

    @property
    def t(self) -> int:
        return max(int(getattr(self.fleet1, "t")), int(getattr(self.fleet2, "t")))

#Centralized product MDP for two interacting E4 fleet operators.
@dataclass
class TwoFleetSharedRoadWorld:
    e4: Any
    fleet1: Any
    fleet2: Any
    cache_edges: bool = True
    max_joint_actions_per_state: Optional[int] = None
    cross_context_actions: bool = True
    charge_vehicle_collision_externality_to_both_bikes: bool = True

    _edge_cache: Dict[
        Tuple[TwoFleetState, TwoFleetAction],
        Tuple[Dict[TwoFleetState, float], np.ndarray],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for attr in ("road_len", "horizon"):
            left = int(getattr(self.fleet1, attr))
            right = int(getattr(self.fleet2, attr))
            if left != right:
                raise ValueError(f"fleet worlds must have the same {attr}: {left} != {right}")

        if bool(getattr(self.fleet1, "stochastic")) != bool(
            getattr(self.fleet2, "stochastic")
        ):
            raise ValueError("fleet worlds must use the same stochastic flag")

        if abs(
            float(getattr(self.fleet1, "slip_prob"))
            - float(getattr(self.fleet2, "slip_prob"))
        ) > 1e-15:
            raise ValueError("fleet worlds must use the same slip probability")
        if int(self.fleet1.start_state.t) != int(self.fleet2.start_state.t):
            raise ValueError("fleet start times must match")

        overlap = self._initial_cross_overlap(self.start_state)
        if overlap:
            raise ValueError(
                "The two copied fleets overlap in the initial shared-road state: "
                + ", ".join(overlap)
                + ". Stagger the fleet-2 start positions."
            )
    @property
    def road_len(self) -> int:
        return int(self.fleet1.road_len)

    @property
    def horizon(self) -> int:
        return int(self.fleet1.horizon)

    @property
    def target_pos(self) -> int:
        return int(self.fleet1.target_pos)

    @property
    def start_state(self) -> TwoFleetState:
        return TwoFleetState(self.fleet1.start_state, self.fleet2.start_state)

    @property
    def role_names(self) -> Tuple[str, ...]:
        return (
            "op1_lead_car",
            "op1_merging_car",
            "op1_emergency_vehicle",
            "op1_cyclist",
            "op2_lead_car",
            "op2_merging_car",
            "op2_emergency_vehicle",
            "op2_cyclist",
        )

    @property
    def operator_names(self) -> Tuple[str, str]:
        return ("operator1", "operator2")

    @property
    def property_names(self) -> Tuple[str, ...]:
        return (
            # Global, non-bargainable safety properties.
            "no_collision",
            "no_cyclist_collision",

            # Operator-specific modifiable norms.
            "op1_c_no_speeding",
            "op1_safe_by_4ft_definition",
            "op1_safe_by_3ft_definition",
            "op1_safe_by_2ft_definition",
            "op2_c_no_speeding",
            "op2_safe_by_4ft_definition",
            "op2_safe_by_3ft_definition",
            "op2_safe_by_2ft_definition",

            # Useful behavioral outcomes for plots.
            "op1_a_on_time",
            "op1_b_on_time",
            "op1_c_on_time",
            "op1_d_on_time",
            "op2_a_on_time",
            "op2_b_on_time",
            "op2_c_on_time",
            "op2_d_on_time",
            "op1_no_collision",
            "op2_no_collision",
        )

    @staticmethod
    def _all_done(local_state: Any) -> bool:
        return bool(
            local_state.a_done
            and local_state.b_done
            and local_state.c_done
            and local_state.d_done
        )

    @staticmethod
    def _local_failure(local_state: Any) -> bool:
        return bool(local_state.collision_bad or local_state.cyclist_collision_bad)

    def is_terminal(self, state: TwoFleetState) -> bool:
        if int(state.t) >= self.horizon:
            return True
        if self._local_failure(state.fleet1) or self._local_failure(state.fleet2):
            return True
        if (
            state.cross_vehicle_collision_bad
            or state.bike1_collision_bad
            or state.bike2_collision_bad
        ):
            return True
        return bool(self._all_done(state.fleet1) and self._all_done(state.fleet2))

    def _held_done_state(self, local_state: Any) -> Any:
        if not self._all_done(local_state):
            raise ValueError("only a completed local fleet can take the product no-op")
        return replace(local_state, t=min(self.horizon, int(local_state.t) + 1))

    def _component_actions(
        self,
        env: Any,
        local_state: Any,
        other_state: Any,
    ) -> List[Optional[Any]]:
        if self._all_done(local_state) and int(local_state.t) < self.horizon:
            return [None]

        if not self.cross_context_actions:
            actions = list(env.actions_from(local_state))
        else:
            CENTER = int(env.e4.CENTER)
            a_speeds = (0,) if local_state.a_done else tuple(int(v) for v in env.a_speeds)
            b_speeds = (0,) if local_state.b_done else tuple(int(v) for v in env.b_speeds)
            c_speeds = (0,) if local_state.c_done else tuple(int(v) for v in env.c_speeds)
            d_speeds = (0,) if local_state.d_done else tuple(int(v) for v in env.d_speeds)

            if local_state.a_done:
                a_lanes = (CENTER,)
            else:
                gaps_behind: List[int] = []
                for role in ("B", "C"):
                    pos, _lane, done = self._actor(local_state, role)
                    if not done and pos < int(local_state.a_pos):
                        gaps_behind.append(int(local_state.a_pos) - pos)
                for role in ("A", "B", "C"):
                    pos, _lane, done = self._actor(other_state, role)
                    if not done and pos < int(local_state.a_pos):
                        gaps_behind.append(int(local_state.a_pos) - pos)
                near_yield_request = bool(gaps_behind) and min(gaps_behind) <= int(env.yield_window)
                a_lanes = (
                    env.negotiable_vehicle_lanes
                    if near_yield_request or int(local_state.a_lane) != CENTER
                    else (CENTER,)
                )

            if local_state.c_done:
                c_lanes = (CENTER,)
            else:
                gaps_ahead: List[int] = []
                for role in ("A", "B", "D"):
                    pos, _lane, done = self._actor(local_state, role)
                    if not done and int(local_state.c_pos) < pos:
                        gaps_ahead.append(pos - int(local_state.c_pos))
                for role in ("A", "B", "C", "D"):
                    pos, _lane, done = self._actor(other_state, role)
                    if not done and int(local_state.c_pos) < pos:
                        gaps_ahead.append(pos - int(local_state.c_pos))
                near_blocker = bool(gaps_ahead) and min(gaps_ahead) <= int(env.ambulance_pass_window)
                c_lanes = (
                    env.negotiable_vehicle_lanes
                    if near_blocker or int(local_state.c_lane) != CENTER
                    else (CENTER,)
                )

            actions = [
                ((int(sa), int(la)), int(sb), (int(sc), int(lc)), int(sd))
                for sa, la, sb, sc, lc, sd in product(
                    a_speeds, a_lanes, b_speeds, c_speeds, c_lanes, d_speeds
                )
            ]

        if not actions and int(local_state.t) < self.horizon:
            raise RuntimeError(
                "A non-completed local component has no actions inside a nonterminal "
                "product state."
            )
        return actions

    def actions_from(self, state: TwoFleetState) -> List[TwoFleetAction]:
        if self.is_terminal(state):
            return []
        a1 = self._component_actions(self.fleet1, state.fleet1, state.fleet2)
        a2 = self._component_actions(self.fleet2, state.fleet2, state.fleet1)
        count = len(a1) * len(a2)
        if (
            self.max_joint_actions_per_state is not None
            and count > int(self.max_joint_actions_per_state)
        ):
            raise RuntimeError(
                f"joint action count {count:,} exceeds "
                f"max_joint_actions_per_state={self.max_joint_actions_per_state:,}; "
                "reduce local action sets instead of silently truncating them"
            )
        return [(left, right) for left, right in product(a1, a2)]

    @staticmethod
    def _actor(local_state: Any, role: str) -> Tuple[int, int, bool]:
        role = role.upper()
        if role == "A":
            return int(local_state.a_pos), int(local_state.a_lane), bool(local_state.a_done)
        if role == "B":
            return int(local_state.b_pos), int(local_state.b_lane), bool(local_state.b_done)
        if role == "C":
            return int(local_state.c_pos), int(local_state.c_lane), bool(local_state.c_done)
        if role == "D":
            return int(local_state.d_pos), int(local_state.d_lane), bool(local_state.d_done)
        raise KeyError(role)

    @staticmethod
    def _crossed(
        old1: int,
        new1: int,
        old2: int,
        new2: int,
    ) -> bool:
        return bool(
            (int(old1) < int(old2) and int(new1) >= int(new2))
            or (int(old2) < int(old1) and int(new2) >= int(new1))
        )

    @staticmethod
    def _interval_hits(old_pos: int, new_pos: int, positions: Iterable[int]) -> bool:
        lo, hi = sorted((int(old_pos), int(new_pos)))
        return any(lo <= int(p) <= hi for p in positions)

    @staticmethod
    def _intervals_near(
        a0: int,
        a1: int,
        b0: int,
        b1: int,
        radius: int = 0,
    ) -> bool:
        alo, ahi = sorted((int(a0), int(a1)))
        blo, bhi = sorted((int(b0), int(b1)))
        return not (ahi < blo - int(radius) or bhi < alo - int(radius))

    def _initial_cross_overlap(self, state: TwoFleetState) -> List[str]:
        overlaps: List[str] = []
        for r1 in ("A", "B", "C"):
            p1, l1, d1 = self._actor(state.fleet1, r1)
            if d1:
                continue
            for r2 in ("A", "B", "C"):
                p2, l2, d2 = self._actor(state.fleet2, r2)
                if not d2 and p1 == p2 and l1 == l2:
                    overlaps.append(f"{r1}1/{r2}2 at (pos={p1}, lane={l1})")
        d1p, d1l, d1done = self._actor(state.fleet1, "D")
        d2p, d2l, d2done = self._actor(state.fleet2, "D")
        if not d1done and not d2done and d1p == d2p and d1l == d2l:
            overlaps.append(f"Bike1/Bike2 at (pos={d1p}, lane={d1l})")
        return overlaps

    def _cross_vehicle_collision_pairs(
        self,
        old: TwoFleetState,
        new1: Any,
        new2: Any,
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for role1 in ("A", "B", "C"):
            old1, _old_lane1, old_done1 = self._actor(old.fleet1, role1)
            new_pos1, new_lane1, new_done1 = self._actor(new1, role1)
            active1 = (not old_done1) and (not new_done1)
            if not active1:
                continue
            for role2 in ("A", "B", "C"):
                old2, _old_lane2, old_done2 = self._actor(old.fleet2, role2)
                new_pos2, new_lane2, new_done2 = self._actor(new2, role2)
                active2 = (not old_done2) and (not new_done2)
                if not active2 or int(new_lane1) != int(new_lane2):
                    continue
                same_final_cell = int(new_pos1) == int(new_pos2)
                crossed = self._crossed(old1, new_pos1, old2, new_pos2)
                if same_final_cell or crossed:
                    pairs.append((role1, role2))
        return pairs

    def _bike_bike_collision(
        self,
        old: TwoFleetState,
        new1: Any,
        new2: Any,
    ) -> bool:
        d1_old, lane1_old, done1_old = self._actor(old.fleet1, "D")
        d2_old, lane2_old, done2_old = self._actor(old.fleet2, "D")
        d1_new, lane1_new, done1_new = self._actor(new1, "D")
        d2_new, lane2_new, done2_new = self._actor(new2, "D")
        active1 = (not done1_old) and (not done1_new)
        active2 = (not done2_old) and (not done2_new)
        if not active1 or not active2 or lane1_new != lane2_new:
            return False
        return bool(
            d1_new == d2_new or self._crossed(d1_old, d1_new, d2_old, d2_new)
        )

    def _cross_pass_for_one_bike(
        self,
        *,
        vehicle_env: Any,
        vehicle_old_state: Any,
        vehicle_new_state: Any,
        bike_env: Any,
        bike_old_state: Any,
        bike_new_state: Any,
    ) -> Tuple[bool, bool, bool, List[str]]:
        curb = False
        close = False
        collision = False
        colliders: List[str] = []

        bike_old, _bike_old_lane, bike_old_done = self._actor(bike_old_state, "D")
        bike_new, _bike_new_lane, bike_new_done = self._actor(bike_new_state, "D")
        bike_active = (not bike_old_done) and (not bike_new_done)
        if bike_old_done:
            return False, False, False, colliders

        dynamic_positions = set(bike_env.contested_positions_at(bike_old))
        dynamic_positions.update(bike_env.contested_positions_at(bike_new))
        for role in ("A", "C"):
            old_pos, _old_lane, old_done = self._actor(vehicle_old_state, role)
            new_pos, new_lane, _new_done = self._actor(vehicle_new_state, role)
            if old_done or not self._interval_hits(old_pos, new_pos, dynamic_positions):
                continue

            category = vehicle_env._lane_clearance_category(int(new_lane))
            if category == "strict_only_3to4ft":
                curb = True
            elif category == "relaxed_unsafe_2to3ft":
                close = True
            elif category == "hard_unsafe_lt2ft":
                close = True
                if bike_active and self._intervals_near(
                    old_pos, new_pos, bike_old, bike_new, radius=0
                ):
                    collision = True
                    colliders.append(role)

        return curb, close, collision, colliders

    @staticmethod
    def _role_index(operator: int, role: str) -> int:
        offset = 0 if int(operator) == 1 else 4
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return offset + mapping[str(role).upper()]

    @staticmethod
    def _vehicle_collision_cost(costs: Any, role: str) -> float:
        return float(getattr(costs, f"{str(role).upper()}_collision"))

    @staticmethod
    def _vehicle_cyclist_collision_cost(costs: Any, role: str) -> float:
        role = str(role).upper()
        name = f"{role}_cyclist_collision"
        if hasattr(costs, name):
            return float(getattr(costs, name))
        # Bob does not enter cyclist clearance bands in the current E4 model.
        return float(getattr(costs, f"{role}_collision"))

    def _add_terminal_late_penalties(
        self,
        extra: np.ndarray,
        *,
        operator: int,
        env: Any,
        successor: Any,
    ) -> None:

        offset = 0 if int(operator) == 1 else 4
        costs = env.costs
        if not successor.a_done:
            extra[offset + 0] += float(costs.A_terminal_late)
        if not env.bob_is_merged(successor):
            extra[offset + 1] += float(costs.B_terminal_late)
        if not successor.c_done:
            extra[offset + 2] += float(costs.C_terminal_late)
        if not successor.d_done:
            extra[offset + 3] += float(getattr(costs, "D_terminal_late", 0.0))

    def _cross_events_and_costs(
        self,
        old: TwoFleetState,
        new1: Any,
        new2: Any,
    ) -> Tuple[Dict[str, bool], np.ndarray]:
        extra = np.zeros(8, dtype=float)

        vehicle_pairs = self._cross_vehicle_collision_pairs(old, new1, new2)
        bike_bike = self._bike_bike_collision(old, new1, new2)

        # Fleet-2 vehicles passing Bike_1.
        b1_curb, b1_close, b1_collision, b1_colliders = self._cross_pass_for_one_bike(
            vehicle_env=self.fleet2,
            vehicle_old_state=old.fleet2,
            vehicle_new_state=new2,
            bike_env=self.fleet1,
            bike_old_state=old.fleet1,
            bike_new_state=new1,
        )

        # Fleet-1 vehicles passing Bike_2.
        b2_curb, b2_close, b2_collision, b2_colliders = self._cross_pass_for_one_bike(
            vehicle_env=self.fleet1,
            vehicle_old_state=old.fleet1,
            vehicle_new_state=new1,
            bike_env=self.fleet2,
            bike_old_state=old.fleet2,
            bike_new_state=new2,
        )

        # Cross vehicle--vehicle collision penalties.
        for role1, role2 in vehicle_pairs:
            extra[self._role_index(1, role1)] += self._vehicle_collision_cost(
                self.fleet1.costs, role1
            )
            extra[self._role_index(2, role2)] += self._vehicle_collision_cost(
                self.fleet2.costs, role2
            )
        if vehicle_pairs and self.charge_vehicle_collision_externality_to_both_bikes:
            extra[self._role_index(1, "D")] += float(
                self.fleet1.costs.D_vehicle_collision_externality
            )
            extra[self._role_index(2, "D")] += float(
                self.fleet2.costs.D_vehicle_collision_externality
            )

        if b1_curb:
            extra[self._role_index(1, "D")] += float(self.fleet1.costs.D_curb_band)
        if b1_close:
            extra[self._role_index(1, "D")] += float(self.fleet1.costs.D_close_pass)
        if b2_curb:
            extra[self._role_index(2, "D")] += float(self.fleet2.costs.D_curb_band)
        if b2_close:
            extra[self._role_index(2, "D")] += float(self.fleet2.costs.D_close_pass)

        # Cross vehicle--bike collision penalties.
        if b1_collision:
            extra[self._role_index(1, "D")] += float(self.fleet1.costs.D_collision)
            for role in b1_colliders:
                extra[self._role_index(2, role)] += self._vehicle_cyclist_collision_cost(
                    self.fleet2.costs, role
                )
        if b2_collision:
            extra[self._role_index(2, "D")] += float(self.fleet2.costs.D_collision)
            for role in b2_colliders:
                extra[self._role_index(1, role)] += self._vehicle_cyclist_collision_cost(
                    self.fleet1.costs, role
                )

        if bike_bike:
            extra[self._role_index(1, "D")] += float(self.fleet1.costs.D_collision)
            extra[self._role_index(2, "D")] += float(self.fleet2.costs.D_collision)
            b1_collision = True
            b2_collision = True

        cross_terminal = bool(vehicle_pairs or b1_collision or b2_collision)
        if cross_terminal:
            if not self.fleet1.is_terminal(new1):
                self._add_terminal_late_penalties(
                    extra, operator=1, env=self.fleet1, successor=new1
                )
            if not self.fleet2.is_terminal(new2):
                self._add_terminal_late_penalties(
                    extra, operator=2, env=self.fleet2, successor=new2
                )

        flags = {
            "cross_vehicle_collision": bool(vehicle_pairs),
            "bike1_curb": bool(b1_curb),
            "bike1_close": bool(b1_close),
            "bike1_collision": bool(b1_collision),
            "bike2_curb": bool(b2_curb),
            "bike2_close": bool(b2_close),
            "bike2_collision": bool(b2_collision),
        }
        return flags, extra

    def _component_transition(
        self,
        env: Any,
        state: Any,
        action: Optional[Any],
    ) -> Tuple[Dict[Any, float], np.ndarray]:
        if action is None:
            return {self._held_done_state(state): 1.0}, np.zeros(4, dtype=float)
        return env.transitions_and_costs(state, action)

    def transitions_and_costs(
        self,
        state: TwoFleetState,
        action: TwoFleetAction,
    ) -> Tuple[Dict[TwoFleetState, float], np.ndarray]:
        if self.is_terminal(state):
            return {}, np.zeros(8, dtype=float)

        cache_key = (state, action)
        if self.cache_edges and cache_key in self._edge_cache:
            trans, costs = self._edge_cache[cache_key]
            return dict(trans), costs.copy()

        trans1, costs1 = self._component_transition(
            self.fleet1, state.fleet1, action[0]
        )
        trans2, costs2 = self._component_transition(
            self.fleet2, state.fleet2, action[1]
        )

        base_costs = np.concatenate(
            [np.asarray(costs1, dtype=float).reshape(4), np.asarray(costs2, dtype=float).reshape(4)]
        )
        expected_extra = np.zeros(8, dtype=float)
        out: Dict[TwoFleetState, float] = {}

        for next1, p1 in trans1.items():
            for next2, p2 in trans2.items():
                p = float(p1) * float(p2)
                if p <= 0.0:
                    continue
                if int(next1.t) != int(next2.t):
                    raise RuntimeError(
                        f"product time desynchronization: {next1.t} != {next2.t}"
                    )

                flags, extra = self._cross_events_and_costs(state, next1, next2)
                successor = TwoFleetState(
                    fleet1=next1,
                    fleet2=next2,
                    cross_vehicle_collision_bad=bool(
                        state.cross_vehicle_collision_bad
                        or flags["cross_vehicle_collision"]
                    ),
                    bike1_curb_band_bad=bool(
                        state.bike1_curb_band_bad or flags["bike1_curb"]
                    ),
                    bike1_close_pass_bad=bool(
                        state.bike1_close_pass_bad or flags["bike1_close"]
                    ),
                    bike1_collision_bad=bool(
                        state.bike1_collision_bad or flags["bike1_collision"]
                    ),
                    bike2_curb_band_bad=bool(
                        state.bike2_curb_band_bad or flags["bike2_curb"]
                    ),
                    bike2_close_pass_bad=bool(
                        state.bike2_close_pass_bad or flags["bike2_close"]
                    ),
                    bike2_collision_bad=bool(
                        state.bike2_collision_bad or flags["bike2_collision"]
                    ),
                )
                out[successor] = out.get(successor, 0.0) + p
                expected_extra += p * extra

        total = float(sum(out.values()))
        if total <= 0.0:
            raise RuntimeError("joint transition distribution is empty")
        if abs(total - 1.0) > 1e-12:
            for key in list(out):
                out[key] /= total
            expected_extra /= total

        costs = base_costs + expected_extra
        if self.cache_edges:
            self._edge_cache[cache_key] = (dict(out), costs.copy())
        return out, costs

    def transitions(
        self,
        state: TwoFleetState,
        action: TwoFleetAction,
    ) -> Dict[TwoFleetState, float]:
        return self.transitions_and_costs(state, action)[0]

    def expected_costs8(
        self,
        state: TwoFleetState,
        action: TwoFleetAction,
    ) -> np.ndarray:
        return self.transitions_and_costs(state, action)[1]

    # PCTL properties
    def _operator_safe_definition(
        self,
        state: TwoFleetState,
        operator: int,
        distance: int,
    ) -> float:
        local = state.fleet1 if int(operator) == 1 else state.fleet2
        env = self.fleet1 if int(operator) == 1 else self.fleet2
        if int(operator) == 1:
            cross_curb = state.bike1_curb_band_bad
            cross_close = state.bike1_close_pass_bad
            cross_collision = state.bike1_collision_bad
        else:
            cross_curb = state.bike2_curb_band_bad
            cross_close = state.bike2_close_pass_bad
            cross_collision = state.bike2_collision_bad

        if int(distance) == 4:
            local_ok = bool(env.property_value(local, "safe_by_4ft_definition"))
            cross_ok = not (cross_curb or cross_close or cross_collision)
        elif int(distance) == 3:
            local_ok = bool(env.property_value(local, "safe_by_3ft_definition"))
            cross_ok = not (cross_close or cross_collision)
        elif int(distance) == 2:
            local_ok = bool(env.property_value(local, "safe_by_2ft_definition"))
            cross_ok = not cross_collision
        else:
            raise ValueError("distance must be 4, 3, or 2")
        return float(local_ok and cross_ok)

    def property_value(self, state: TwoFleetState, name: str) -> float:
        if name == "no_collision":
            ok = (
                not state.fleet1.collision_bad
                and not state.fleet2.collision_bad
                and not state.cross_vehicle_collision_bad
            )
            return float(ok)
        if name == "no_cyclist_collision":
            ok = (
                not state.fleet1.cyclist_collision_bad
                and not state.fleet2.cyclist_collision_bad
                and not state.bike1_collision_bad
                and not state.bike2_collision_bad
            )
            return float(ok)

        for op, prefix, local, env in (
            (1, "op1", state.fleet1, self.fleet1),
            (2, "op2", state.fleet2, self.fleet2),
        ):
            if name == f"{prefix}_c_no_speeding":
                return float(env.property_value(local, "c_no_speeding"))
            if name == f"{prefix}_safe_by_4ft_definition":
                return self._operator_safe_definition(state, op, 4)
            if name == f"{prefix}_safe_by_3ft_definition":
                return self._operator_safe_definition(state, op, 3)
            if name == f"{prefix}_safe_by_2ft_definition":
                return self._operator_safe_definition(state, op, 2)
            if name == f"{prefix}_a_on_time":
                return float(env.property_value(local, "a_on_time"))
            if name == f"{prefix}_b_on_time":
                return float(env.property_value(local, "b_on_time"))
            if name == f"{prefix}_c_on_time":
                return float(env.property_value(local, "c_on_time"))
            if name == f"{prefix}_d_on_time":
                return float(env.property_value(local, "d_on_time"))
            if name == f"{prefix}_no_collision":
                return float(
                    (not local.collision_bad)
                    and (not state.cross_vehicle_collision_bad)
                )

        raise KeyError(f"Unknown two-fleet property {name!r}")

    # Reachability and presentation
    def reachable_states(
        self,
        *,
        max_states: int = 100_000,
        verbose: bool = False,
    ) -> List[TwoFleetState]:
        seen = {self.start_state}
        frontier = [self.start_state]
        expanded = 0
        while frontier:
            state = frontier.pop()
            if self.is_terminal(state):
                continue
            expanded += 1
            for action in self.actions_from(state):
                for successor in self.transitions(state, action):
                    if successor not in seen:
                        seen.add(successor)
                        frontier.append(successor)
                        if len(seen) > int(max_states):
                            raise RuntimeError(
                                f"two-fleet reachable state count exceeded "
                                f"max_states={max_states:,}"
                            )
            if verbose and expanded % 100 == 0:
                print(
                    f"[two-fleet graph] expanded={expanded:,}, "
                    f"discovered={len(seen):,}"
                )
        return sorted(seen)

    def format_action(self, action: TwoFleetAction) -> str:
        left = "DONE/no-op" if action[0] is None else self.fleet1.format_action(action[0])
        right = "DONE/no-op" if action[1] is None else self.fleet2.format_action(action[1])
        return f"Fleet1[{left}] || Fleet2[{right}]"

    def ascii_map(self, state: Optional[TwoFleetState] = None) -> str:
        state = self.start_state if state is None else state
        lane_ids = sorted(int(k) for k in self.e4.LANE_NAMES)
        rows: Dict[int, List[str]] = {
            lane: ["·"] * self.road_len for lane in lane_ids
        }
        for lane in lane_ids:
            rows[lane][self.target_pos] = "T"

        def place(lane: int, pos: int, symbol: str) -> None:
            lane = int(lane)
            if lane not in rows:
                rows[lane] = ["·"] * self.road_len
                rows[lane][self.target_pos] = "T"
            current = rows[lane][int(pos)]
            rows[lane][int(pos)] = symbol if current in {"·", "T"} else current + "/" + symbol

        for op, local in ((1, state.fleet1), (2, state.fleet2)):
            if not local.a_done:
                place(local.a_lane, local.a_pos, f"A{op}")
            if not local.b_done:
                place(local.b_lane, local.b_pos, f"B{op}")
            if not local.c_done:
                place(local.c_lane, local.c_pos, f"E{op}")
            if not local.d_done:
                place(local.d_lane, local.d_pos, f"K{op}")

        lines = []
        for lane in lane_ids:
            name = str(self.e4.LANE_NAMES.get(lane, lane))
            lines.append(f"{name:24s}: " + " ".join(rows[lane]))
        lines.append(
            "flags: "
            f"cross_vehicle_collision={state.cross_vehicle_collision_bad}, "
            f"Bike1(curb={state.bike1_curb_band_bad}, close={state.bike1_close_pass_bad}, "
            f"collision={state.bike1_collision_bad}), "
            f"Bike2(curb={state.bike2_curb_band_bad}, close={state.bike2_close_pass_bad}, "
            f"collision={state.bike2_collision_bad})"
        )
        lines.append(
            "Legend: A=lead car, B=merging car, E=emergency vehicle, "
            "K=cyclist; suffix is operator number."
        )
        return "\n".join(lines)

@dataclass
class TwoFleetLPData:
    env: TwoFleetSharedRoadWorld
    states: List[TwoFleetState]
    nonterminal_states: List[TwoFleetState]
    state_index: Dict[TwoFleetState, int]
    edges: List[Tuple[TwoFleetState, TwoFleetAction]]
    A_flow: sp.csr_matrix
    b_flow: np.ndarray
    C_roles: np.ndarray
    C_operators: np.ndarray
    q: Dict[str, np.ndarray]
    role_names: Tuple[str, ...]
    operator_names: Tuple[str, str]
    operator_internal_weights: np.ndarray
    build_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def E(self) -> int:
        return len(self.edges)

    @property
    def C(self) -> np.ndarray:
        return self.C_operators


@dataclass(frozen=True)
class SharedTwoFleetRelaxation:
    row_names: Tuple[str, ...]
    z_names: Tuple[str, str, str, str]
    G_relax: np.ndarray
    b_relax: np.ndarray
    slack_map: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        G = np.asarray(self.G_relax, dtype=float)
        b = np.asarray(self.b_relax, dtype=float).reshape(-1)
        R = np.asarray(self.slack_map, dtype=float)
        if G.ndim != 2:
            raise ValueError("G_relax must be a matrix")
        if b.shape != (G.shape[0],):
            raise ValueError("b_relax length must match G_relax rows")
        if R.shape != (G.shape[0], 4):
            raise ValueError("slack_map must have shape (num_rows, 4)")
        if len(self.row_names) != G.shape[0]:
            raise ValueError("row_names length must match G_relax rows")
        object.__setattr__(self, "G_relax", G)
        object.__setattr__(self, "b_relax", b)
        object.__setattr__(self, "slack_map", R)

    @property
    def num_rows(self) -> int:
        return int(self.G_relax.shape[0])


@dataclass
class TwoFleetBargainingProblem:
    lp_data: TwoFleetLPData
    relaxation: SharedTwoFleetRelaxation
    hard_thresholds: Dict[str, float]
    z_max: np.ndarray
    semantic_tolerance: float = 0.01
    numerical_tolerance: float = 1e-7

    @property
    def operator_costs(self) -> np.ndarray:
        return np.asarray(self.lp_data.C_operators, dtype=float)

    @property
    def operator_names(self) -> Tuple[str, str]:
        return self.lp_data.operator_names


def _normalized_weights(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(tuple(values), dtype=float).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must contain four role weights")
    if np.any(arr < 0.0) or float(arr.sum()) <= 0.0:
        raise ValueError(f"{name} must be nonnegative and not all zero")
    return arr / float(arr.sum())


def build_two_fleet_lp_data(
    env: TwoFleetSharedRoadWorld,
    *,
    operator1_internal_omega: Sequence[float] = (0.10, 0.10, 0.60, 0.20),
    operator2_internal_omega: Sequence[float] = (0.10, 0.10, 0.30, 0.50),
    max_states: int = 100_000,
    verbose: bool = False,
) -> TwoFleetLPData:
    """Build the joint occupation-measure LP data with eight role costs."""

    total_t0 = time.perf_counter()
    w1 = _normalized_weights(operator1_internal_omega, name="operator1_internal_omega")
    w2 = _normalized_weights(operator2_internal_omega, name="operator2_internal_omega")

    graph_t0 = time.perf_counter()
    states = env.reachable_states(max_states=max_states, verbose=verbose)
    nonterminal = [state for state in states if not env.is_terminal(state)]
    state_index = {state: i for i, state in enumerate(nonterminal)}
    edges: List[Tuple[TwoFleetState, TwoFleetAction]] = []
    for state in nonterminal:
        for action in env.actions_from(state):
            edges.append((state, action))
    graph_s = time.perf_counter() - graph_t0

    matrix_t0 = time.perf_counter()
    S = len(nonterminal)
    E = len(edges)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b = np.zeros(S, dtype=float)
    start_idx = state_index.get(env.start_state)
    if start_idx is None:
        raise RuntimeError("joint start state is terminal or absent")
    b[start_idx] = 1.0

    C_roles = np.zeros((8, E), dtype=float)
    q = {name: np.zeros(E, dtype=float) for name in env.property_names}

    for edge_index, (state, action) in enumerate(edges):
        i_from = state_index[state]
        rows.append(i_from)
        cols.append(edge_index)
        vals.append(1.0)

        transitions, costs8 = env.transitions_and_costs(state, action)
        C_roles[:, edge_index] = np.asarray(costs8, dtype=float).reshape(8)

        for successor, probability_raw in transitions.items():
            probability = float(probability_raw)
            if env.is_terminal(successor):
                for prop in env.property_names:
                    q[prop][edge_index] += probability * env.property_value(successor, prop)
            else:
                i_to = state_index[successor]
                rows.append(i_to)
                cols.append(edge_index)
                vals.append(-probability)

    A_flow = sp.csr_matrix((vals, (rows, cols)), shape=(S, E))
    C_operators = np.vstack([w1 @ C_roles[:4, :], w2 @ C_roles[4:, :]])
    matrix_s = time.perf_counter() - matrix_t0
    total_s = time.perf_counter() - total_t0

    metrics = {
        "graph_construction_s": float(graph_s),
        "occupation_matrix_construction_s": float(matrix_s),
        "total_lp_data_build_s": float(total_s),
        "max_joint_actions_observed": float(
            max((len(env.actions_from(s)) for s in nonterminal), default=0)
        ),
    }
    if verbose:
        print(
            f"[build_two_fleet_lp_data] states={len(states):,}, "
            f"nonterminal={S:,}, edges={E:,}, flow_nnz={A_flow.nnz:,}, "
            f"build={total_s:.3f}s"
        )

    return TwoFleetLPData(
        env=env,
        states=states,
        nonterminal_states=nonterminal,
        state_index=state_index,
        edges=edges,
        A_flow=A_flow,
        b_flow=b,
        C_roles=C_roles,
        C_operators=C_operators,
        q=q,
        role_names=env.role_names,
        operator_names=env.operator_names,
        operator_internal_weights=np.vstack([w1, w2]),
        build_metrics=metrics,
    )


def build_shared_two_fleet_relaxation(
    lp_data: TwoFleetLPData,
    *,
    theta_speed: float = 1.0,
    lambda_definition: float = 1.0,
) -> SharedTwoFleetRelaxation:

    z_names = ("tau", "sigma_4ft", "sigma_3ft", "sigma_2ft")
    rows: List[np.ndarray] = []
    b: List[float] = []
    row_names: List[str] = []
    R: List[np.ndarray] = []

    for prefix in ("op1", "op2"):
        properties = (
            f"{prefix}_c_no_speeding",
            f"{prefix}_safe_by_4ft_definition",
            f"{prefix}_safe_by_3ft_definition",
            f"{prefix}_safe_by_2ft_definition",
        )
        missing = [prop for prop in properties if prop not in lp_data.q]
        if missing:
            raise KeyError(f"missing two-fleet norm properties: {missing}")
        thresholds = (
            float(theta_speed),
            float(lambda_definition),
            float(lambda_definition),
            float(lambda_definition),
        )
        for j, (prop, threshold) in enumerate(zip(properties, thresholds)):
            rows.append(-np.asarray(lp_data.q[prop], dtype=float).reshape(-1))
            b.append(threshold)
            row_names.append(f"{prefix}_{z_names[j]}")
            map_row = np.zeros(4, dtype=float)
            map_row[j] = 1.0
            R.append(map_row)

    return SharedTwoFleetRelaxation(
        row_names=tuple(row_names),
        z_names=z_names,
        G_relax=np.vstack(rows),
        b_relax=np.asarray(b, dtype=float),
        slack_map=np.vstack(R),
        metadata={
            "interpretation": (
                "The shared norm is applied separately to both fleet operators: "
                "Gx+b <= [I4; I4]z."
            ),
            "theta_speed": float(theta_speed),
            "lambda_definition": float(lambda_definition),
            "sigma_order": "sigma_2ft <= sigma_3ft <= sigma_4ft",
        },
    )


def build_two_fleet_bargaining_problem(
    env: TwoFleetSharedRoadWorld,
    *,
    operator1_internal_omega: Sequence[float] = (0.10, 0.10, 0.60, 0.20),
    operator2_internal_omega: Sequence[float] = (0.10, 0.10, 0.30, 0.50),
    theta_speed: float = 1.0,
    lambda_definition: float = 1.0,
    hard_safety_threshold: float = 1.0,
    z_max: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    max_states: int = 100_000,
    semantic_tolerance: float = 0.01,
    numerical_tolerance: float = 1e-7,
    verbose: bool = False,
) -> TwoFleetBargainingProblem:
    lp_data = build_two_fleet_lp_data(
        env,
        operator1_internal_omega=operator1_internal_omega,
        operator2_internal_omega=operator2_internal_omega,
        max_states=max_states,
        verbose=verbose,
    )
    relaxation = build_shared_two_fleet_relaxation(
        lp_data,
        theta_speed=theta_speed,
        lambda_definition=lambda_definition,
    )
    caps = np.asarray(tuple(z_max), dtype=float).reshape(-1)
    if caps.shape != (4,) or np.any(caps <= 0.0):
        raise ValueError("z_max must contain four positive entries")
    return TwoFleetBargainingProblem(
        lp_data=lp_data,
        relaxation=relaxation,
        hard_thresholds={
            "no_collision": float(hard_safety_threshold),
            "no_cyclist_collision": float(hard_safety_threshold),
        },
        z_max=caps,
        semantic_tolerance=float(semantic_tolerance),
        numerical_tolerance=float(numerical_tolerance),
    )


def make_default_two_fleet_world(
    e4: Any,
    *,
    horizon: int = 2,
    road_len: int = 12,
    stochastic: bool = False,
    slip_prob: float = 0.0,
    allow_2ft_band: bool = False,
    allow_hard_lt2ft_band: bool = False,
    speed_event_mode: str = "commanded",
    fleet1_overrides: Optional[Mapping[str, Any]] = None,
    fleet2_overrides: Optional[Mapping[str, Any]] = None,
    max_joint_actions_per_state: Optional[int] = 2_000,
) -> TwoFleetSharedRoadWorld:
    if not hasattr(e4, "make_e4_moving_bike_env"):
        raise AttributeError(
            "e4.make_e4_moving_bike_env is missing; install the moving-bike patch first"
        )

    common = {
        "road_len": int(road_len),
        "horizon": int(horizon),
        "stochastic": bool(stochastic),
        "slip_prob": float(slip_prob),
        "speed_event_mode": str(speed_event_mode),
        "allow_2ft_band": bool(allow_2ft_band),
        "allow_hard_lt2ft_band": bool(allow_hard_lt2ft_band),
        "a_speeds": (1,),
        "b_speeds": (1,),
        "c_speeds": (2, 3),
        "d_speeds": (0,),
        "a_speed_limit": 1,
        "b_speed_limit": 1,
        "c_speed_limit": 2,
        "d_speed_limit": 0,
    }

    defaults1 = {
        **common,
        "a_start": 5,
        "b_start": 2,
        "c_start": 0,
        "d_start": min(int(road_len) - 3, 9),
        "b_merge_goal_pos": 7,
    }
    defaults2 = {
        **common,
        "a_start": 7,
        "b_start": 3,
        "c_start": 1,
        "d_start": min(int(road_len) - 2, 10),
        "b_merge_goal_pos": 8,
    }
    defaults1.update(dict(fleet1_overrides or {}))
    defaults2.update(dict(fleet2_overrides or {}))

    fleet1 = e4.make_e4_moving_bike_env(**defaults1)
    fleet2 = e4.make_e4_moving_bike_env(**defaults2)
    return TwoFleetSharedRoadWorld(
        e4=e4,
        fleet1=fleet1,
        fleet2=fleet2,
        max_joint_actions_per_state=max_joint_actions_per_state,
    )


__all__ = [
    "TwoFleetState",
    "TwoFleetSharedRoadWorld",
    "TwoFleetLPData",
    "SharedTwoFleetRelaxation",
    "TwoFleetBargainingProblem",
    "load_e4_with_moving_bike_patch",
    "make_default_two_fleet_world",
    "build_two_fleet_lp_data",
    "build_shared_two_fleet_relaxation",
    "build_two_fleet_bargaining_problem",
]
