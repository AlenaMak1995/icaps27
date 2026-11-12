from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# Lateral distance bands from the cyclist/curb edge.
CYCLIST_EDGE = 0
LT_2FT = 1
FT_2_3 = 2
FT_3_4 = 3
SAFE_LANE = 4
MERGE_LANE = 5
CURB = FT_3_4
CENTER = SAFE_LANE

LANE_NAMES = {
    CYCLIST_EDGE: "cyclist/curb-edge",
    LT_2FT: "<2ft-hard-unsafe",
    FT_2_3: "2-3ft-close-pass",
    FT_3_4: "3-4ft-definition-band",
    SAFE_LANE: ">=4ft-safe-lane",
    MERGE_LANE: "merge-lane",
}


@dataclass
class E4CostParams:
    A_time: float = 0.12
    A_speed_effort: float = 0.04
    A_speeding: float = 28.0
    A_curb: float = 2.50
    A_tailgated: float = 4.00
    A_collision: float = 180.0
    A_cyclist_collision: float = 250.0
    A_terminal_late: float = 8.0

    B_time: float = 2.40
    B_speed_effort: float = 0.03
    B_speeding: float = 0.18
    B_tailgate: float = 1.20
    B_collision: float = 180.0
    B_terminal_late: float = 32.0

    C_time: float = 4.50
    C_speed_effort: float = 0.02
    C_speeding: float = 0.05
    C_curb: float = 0.20
    C_collision: float = 200.0
    C_cyclist_collision: float = 300.0
    C_terminal_late: float = 70.0

    D_close_pass: float = 8.00
    D_curb_band: float = 2.00
    D_collision: float = 500.0
    D_vehicle_collision_externality: float = 15.0


# ---------------------------------------------------------------------------
# 1) Moving-bike E4 environment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class E4BikeState:
    t: int
    a_pos: int
    a_lane: int
    b_pos: int
    b_lane: int
    c_pos: int
    c_lane: int
    d_pos: int
    d_lane: int
    a_done: bool
    b_done: bool
    c_done: bool
    d_done: bool
    a_speed_bad: bool
    b_speed_bad: bool
    c_speed_bad: bool
    collision_bad: bool
    cyclist_collision_bad: bool
    tailgate_count: int
    curb_band_count: int
    close_pass_count: int


@dataclass
class E4MovingBikeWorld:

    e4: Any
    road_len: int = 25
    horizon: int = 10
    a_start: Optional[int] = None
    b_start: int = 1
    c_start: int = 0
    d_start: Optional[int] = None
    b_merge_goal_pos: Optional[int] = None
    a_speed_limit: int = 2
    b_speed_limit: int = 2
    c_speed_limit: int = 2
    d_speed_limit: int = 1
    a_speeds: Tuple[int, ...] = (1, 2)
    b_speeds: Tuple[int, ...] = (1, 2)
    c_speeds: Tuple[int, ...] = (2, 3, 4)
    d_speeds: Tuple[int, ...] = (0, 1)
    yield_window: int = 4
    ambulance_pass_window: int = 6
    slip_prob: float = 0.0
    stochastic: bool = False
    # ``commanded`` preserves the original rule semantics: choosing an illegal
    # action is speeding even if stochastic slip realizes a legal speed.
    # ``realized`` monitors physical sampled speed instead.
    speed_event_mode: str = "commanded"
    tailgate_gap: int = 1
    tailgate_limit: int = 0
    curb_band_limit: int = 1
    close_pass_limit: int = 0
    cyclist_zone_radius: int = 1
    allow_2ft_band: bool = True
    allow_hard_lt2ft_band: bool = False
    costs: Any = None
    cache_edges: bool = True
    _edge_cache: Dict[Tuple[E4BikeState, Tuple[Any, ...]], Tuple[Dict[E4BikeState, float], np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.road_len < 8:
            raise ValueError("road_len must be at least 8")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.a_start is None:
            self.a_start = max(5, int(self.road_len) // 3 + 1)
        if self.d_start is None:
            candidate = max(int(self.a_start) + 4, int(self.road_len) - 6)
            self.d_start = min(int(self.road_len) - 2, candidate)
        if self.b_merge_goal_pos is None:
            self.b_merge_goal_pos = max(self.a_start + 1, int(self.road_len) - 4)
        if self.costs is None:
            self.costs = self.e4.E4CostParams()
        if not hasattr(self.e4, "MERGE_LANE"):
            self.e4.MERGE_LANE = 5
        if not hasattr(self.e4, "LANE_NAMES"):
            self.e4.LANE_NAMES = {}
        if int(self.e4.MERGE_LANE) not in self.e4.LANE_NAMES:
            self.e4.LANE_NAMES[int(self.e4.MERGE_LANE)] = "merge-lane"

        if not (0 <= self.c_start <= self.b_start < int(self.a_start) < self.road_len):
            raise ValueError("expected 0 <= c_start <= b_start < a_start < road_len")
        if not (0 <= int(self.d_start) < self.road_len):
            raise ValueError("d_start must lie inside the road")
        if not (0.0 <= self.slip_prob <= 1.0):
            raise ValueError("slip_prob must lie in [0,1]")
        self.speed_event_mode = str(self.speed_event_mode).strip().lower()
        if self.speed_event_mode not in {"commanded", "realized"}:
            raise ValueError("speed_event_mode must be 'commanded' or 'realized'")

    @property
    def target_pos(self) -> int:
        return int(self.road_len) - 1

    @property
    def cyclist_pos(self) -> int:
        return int(self.d_start)

    @property
    def merge_lane(self) -> int:
        return int(getattr(self.e4, "MERGE_LANE", 5))

    @property
    def safe_lane(self) -> int:
        return int(self.e4.SAFE_LANE)

    def bob_lane_for_position(self, pos: int, current_lane: Optional[int] = None) -> int:
        if current_lane is not None and int(current_lane) == self.safe_lane:
            return self.safe_lane
        return self.safe_lane if int(pos) >= int(self.b_merge_goal_pos) else self.merge_lane

    def bob_is_merged(self, st: E4BikeState) -> bool:
        return bool(st.b_done or int(st.b_lane) == self.safe_lane)

    @property
    def start_state(self) -> E4BikeState:
        return E4BikeState(
            t=0,
            a_pos=int(self.a_start),
            a_lane=int(self.e4.CENTER),
            b_pos=int(self.b_start),
            b_lane=self.bob_lane_for_position(int(self.b_start)),
            c_pos=int(self.c_start),
            c_lane=int(self.e4.CENTER),
            d_pos=int(self.d_start),
            d_lane=int(self.e4.CYCLIST_EDGE),
            a_done=False,
            b_done=False,
            c_done=False,
            d_done=False,
            a_speed_bad=False,
            b_speed_bad=False,
            c_speed_bad=False,
            collision_bad=False,
            cyclist_collision_bad=False,
            tailgate_count=0,
            curb_band_count=0,
            close_pass_count=0,
        )

    @property
    def property_names(self) -> Tuple[str, ...]:
        return (
            "a_on_time",
            # b_on_time: Bob has completed the merge.
            # b_reached_target is the stronger property that Bob reached road end.
            "b_on_time",
            "b_merged",
            "b_reached_target",
            "c_on_time",
            "d_on_time",
            "a_no_speeding",
            "b_no_speeding",
            "c_no_speeding",
            "no_collision",
            "no_cyclist_collision",
            "courtesy_norm",
            "tailgate_ok",
            # Bad-event monitors kept for compatibility.
            "state_definition_relaxation_3to4ft",
            "strict_4ft_close_pass_use",
            "relaxed_3ft_close_pass_use",
            "curb_band_use",
            "close_pass_ok",
            # Good-event candidate definitions for ordered sigma bargaining.
            "safe_by_4ft_definition",
            "safe_by_3ft_definition",
            "safe_by_2ft_definition",
        )

    @property
    def negotiable_vehicle_lanes(self) -> Tuple[int, ...]:
        lanes = [int(self.e4.CENTER), int(self.e4.FT_3_4)]
        if bool(self.allow_2ft_band):
            lanes.append(int(self.e4.FT_2_3))
        if bool(self.allow_hard_lt2ft_band):
            lanes.append(int(self.e4.LT_2FT))
        out: List[int] = []
        for lane in lanes:
            if lane not in out:
                out.append(lane)
        return tuple(out)

    def contested_positions_at(self, d_pos: int) -> Tuple[int, ...]:
        lo = max(0, int(d_pos) - int(self.cyclist_zone_radius))
        hi = min(self.target_pos, int(d_pos) + int(self.cyclist_zone_radius))
        return tuple(range(lo, hi + 1))

    @property
    def contested_curb_positions(self) -> Tuple[int, ...]:
        return self.contested_positions_at(int(self.d_start))

    def is_terminal(self, st: E4BikeState) -> bool:
        return (
            int(st.t) >= int(self.horizon)
            or bool(st.collision_bad)
            or bool(st.cyclist_collision_bad)
            or (bool(st.a_done) and bool(st.b_done) and bool(st.c_done) and bool(st.d_done))
        )

    def actions_from(self, st: E4BikeState) -> List[Tuple[Any, int, Any, int]]:
        if self.is_terminal(st):
            return []

        CENTER = int(self.e4.CENTER)
        a_speeds = (0,) if st.a_done else tuple(int(v) for v in self.a_speeds)
        b_speeds = (0,) if st.b_done else tuple(int(v) for v in self.b_speeds)
        c_speeds = (0,) if st.c_done else tuple(int(v) for v in self.c_speeds)
        d_speeds = (0,) if st.d_done else tuple(int(v) for v in self.d_speeds)

        if st.a_done:
            a_lanes = (CENTER,)
        else:
            b_gap = st.a_pos - st.b_pos if (not st.b_done and st.b_pos < st.a_pos) else math.inf
            c_gap = st.a_pos - st.c_pos if (not st.c_done and st.c_pos < st.a_pos) else math.inf
            a_lanes = self.negotiable_vehicle_lanes if min(b_gap, c_gap) <= int(self.yield_window) else (CENTER,)

        if st.c_done:
            c_lanes = (CENTER,)
        else:
            gaps: List[int] = []
            if not st.a_done and st.c_pos < st.a_pos:
                gaps.append(st.a_pos - st.c_pos)
            if not st.b_done and st.c_pos < st.b_pos:
                gaps.append(st.b_pos - st.c_pos)
            # Also let ER consider lateral movement when it approaches the bike.
            if not st.d_done and st.c_pos < st.d_pos:
                gaps.append(st.d_pos - st.c_pos)
            near_blocker = bool(gaps) and min(gaps) <= int(self.ambulance_pass_window)
            c_lanes = self.negotiable_vehicle_lanes if (near_blocker or st.c_lane != CENTER) else (CENTER,)

        return [
            ((int(sa), int(la)), int(sb), (int(sc), int(lc)), int(sd))
            for sa, la, sb, sc, lc, sd in product(a_speeds, a_lanes, b_speeds, c_speeds, c_lanes, d_speeds)
        ]

    def _actual_speed_dist(self, commanded_speed: int) -> Dict[int, float]:
        commanded_speed = int(commanded_speed)
        if commanded_speed <= 0 or (not self.stochastic) or self.slip_prob <= 0:
            return {max(0, commanded_speed): 1.0}
        slow = max(0, commanded_speed - 1)
        if slow == commanded_speed:
            return {commanded_speed: 1.0}
        return {commanded_speed: 1.0 - float(self.slip_prob), slow: float(self.slip_prob)}

    @staticmethod
    def _cap_counter(value: int, limit: int) -> int:
        return min(int(value), int(limit) + 1)

    @staticmethod
    def _interval_hits(old_pos: int, new_pos: int, positions: Iterable[int]) -> bool:
        lo = min(int(old_pos), int(new_pos))
        hi = max(int(old_pos), int(new_pos))
        return any(lo <= int(x) <= hi for x in positions)

    @staticmethod
    def _intervals_near(a0: int, a1: int, b0: int, b1: int, radius: int = 0) -> bool:
        alo, ahi = min(int(a0), int(a1)), max(int(a0), int(a1))
        blo, bhi = min(int(b0), int(b1)), max(int(b0), int(b1))
        return not (ahi < blo - int(radius) or bhi < alo - int(radius))

    @staticmethod
    def _same_lane_cross_or_overlap(old_back: int, new_back: int, old_front: int, new_front: int) -> bool:
        return int(old_back) < int(old_front) and int(new_back) >= int(new_front)

    def _lane_clearance_category(self, lane: int) -> str:
        lane = int(lane)
        if lane == int(self.e4.FT_3_4):
            return "strict_only_3to4ft"
        if lane == int(self.e4.FT_2_3):
            return "relaxed_unsafe_2to3ft"
        if lane == int(self.e4.LT_2FT):
            return "hard_unsafe_lt2ft"
        return "safe"

    def _one_successor(
        self,
        st: E4BikeState,
        action: Tuple[Any, int, Any, int],
        a_actual_speed: int,
        b_actual_speed: int,
        c_actual_speed: int,
        d_actual_speed: int,
    ) -> Tuple[E4BikeState, Dict[str, float]]:
        CENTER = int(self.e4.CENTER)
        CYCLIST_EDGE = int(self.e4.CYCLIST_EDGE)
        (a_cmd_speed, a_next_lane), b_cmd_speed, (c_cmd_speed, c_next_lane), d_cmd_speed = action

        if st.a_done:
            a_next, a_lane_after, a_done_after = self.target_pos, CENTER, True
        else:
            a_next = min(self.target_pos, int(st.a_pos) + max(0, int(a_actual_speed)))
            a_lane_after = int(a_next_lane)
            a_done_after = a_next >= self.target_pos

        if st.b_done:
            b_next, b_lane_after, b_done_after = self.target_pos, self.safe_lane, True
        else:
            # Bob chooses speed only.  His lane is part of the state and changes
            # deterministically from merge lane to safe lane at the merge point.
            b_next = min(self.target_pos, int(st.b_pos) + max(0, int(b_actual_speed)))
            b_lane_after = self.bob_lane_for_position(b_next, current_lane=st.b_lane)
            b_done_after = b_next >= self.target_pos

        if st.c_done:
            c_next, c_lane_after, c_done_after = self.target_pos, CENTER, True
        else:
            c_next = min(self.target_pos, int(st.c_pos) + max(0, int(c_actual_speed)))
            c_lane_after = int(c_next_lane)
            c_done_after = c_next >= self.target_pos

        if st.d_done:
            d_next, d_done_after = self.target_pos, True
        else:
            d_next = min(self.target_pos, int(st.d_pos) + max(0, int(d_actual_speed)))
            d_done_after = d_next >= self.target_pos

        a_active = (not st.a_done) and (not a_done_after)
        b_active = (not st.b_done) and (not b_done_after)
        c_active = (not st.c_done) and (not c_done_after)
        d_active = (not st.d_done) and (not d_done_after)

        collision_event = False
        if a_active and b_active and a_lane_after == b_lane_after and a_next == b_next:
            collision_event = True
        if a_active and c_active and a_lane_after == c_lane_after and a_next == c_next:
            collision_event = True
        if b_active and c_active and b_lane_after == c_lane_after and b_next == c_next:
            collision_event = True
        if a_active and b_active and a_lane_after == b_lane_after:
            if self._same_lane_cross_or_overlap(st.b_pos, b_next, st.a_pos, a_next):
                collision_event = True
        if a_active and c_active and a_lane_after == c_lane_after:
            if self._same_lane_cross_or_overlap(st.c_pos, c_next, st.a_pos, a_next):
                collision_event = True
        if b_active and c_active and b_lane_after == c_lane_after:
            if self._same_lane_cross_or_overlap(st.c_pos, c_next, st.b_pos, b_next):
                collision_event = True

        tailgate_event = False
        if (not collision_event) and a_active and b_active and a_lane_after == b_lane_after:
            if b_next < a_next and (a_next - b_next) <= int(self.tailgate_gap):
                tailgate_event = True

        curb_band_event = False
        close_pass_event = False
        cyclist_collision_event = False

        dyn_positions = set(self.contested_positions_at(st.d_pos)) | set(self.contested_positions_at(d_next))

        def update_dynamic_cyclist_events(old_pos: int, new_pos: int, lane: int, done_after: bool) -> None:
            nonlocal curb_band_event, close_pass_event, cyclist_collision_event
            if not self._interval_hits(old_pos, new_pos, dyn_positions):
                return
            category = self._lane_clearance_category(lane)
            if category == "strict_only_3to4ft":
                curb_band_event = True
            elif category == "relaxed_unsafe_2to3ft":
                close_pass_event = True
            elif category == "hard_unsafe_lt2ft":
                close_pass_event = True
                if d_active and self._intervals_near(old_pos, new_pos, st.d_pos, d_next, radius=0):
                    cyclist_collision_event = True

        if not st.a_done:
            update_dynamic_cyclist_events(st.a_pos, a_next, a_lane_after, a_done_after)
        if not st.c_done:
            update_dynamic_cyclist_events(st.c_pos, c_next, c_lane_after, c_done_after)

        if self.speed_event_mode == "realized":
            a_speed_for_norm = int(a_actual_speed)
            b_speed_for_norm = int(b_actual_speed)
            c_speed_for_norm = int(c_actual_speed)
        else:
            a_speed_for_norm = int(a_cmd_speed)
            b_speed_for_norm = int(b_cmd_speed)
            c_speed_for_norm = int(c_cmd_speed)

        a_speed_event = (not st.a_done) and a_speed_for_norm > int(self.a_speed_limit)
        b_speed_event = (not st.b_done) and b_speed_for_norm > int(self.b_speed_limit)
        c_speed_event = (not st.c_done) and c_speed_for_norm > int(self.c_speed_limit)

        ns = E4BikeState(
            t=int(st.t) + 1,
            a_pos=int(a_next),
            a_lane=int(a_lane_after),
            b_pos=int(b_next),
            b_lane=int(b_lane_after),
            c_pos=int(c_next),
            c_lane=int(c_lane_after),
            d_pos=int(d_next),
            d_lane=CYCLIST_EDGE,
            a_done=bool(a_done_after),
            b_done=bool(b_done_after),
            c_done=bool(c_done_after),
            d_done=bool(d_done_after),
            a_speed_bad=bool(st.a_speed_bad or a_speed_event),
            b_speed_bad=bool(st.b_speed_bad or b_speed_event),
            c_speed_bad=bool(st.c_speed_bad or c_speed_event),
            collision_bad=bool(st.collision_bad or collision_event),
            cyclist_collision_bad=bool(st.cyclist_collision_bad or cyclist_collision_event),
            tailgate_count=self._cap_counter(int(st.tailgate_count) + int(tailgate_event), self.tailgate_limit),
            curb_band_count=self._cap_counter(int(st.curb_band_count) + int(curb_band_event), self.curb_band_limit),
            close_pass_count=self._cap_counter(int(st.close_pass_count) + int(close_pass_event), self.close_pass_limit),
        )

        return ns, {
            "a_speed_event": float(a_speed_event),
            "b_speed_event": float(b_speed_event),
            "c_speed_event": float(c_speed_event),
            "collision_event": float(collision_event),
            "cyclist_collision_event": float(cyclist_collision_event),
            "tailgate_event": float(tailgate_event),
            "curb_band_event": float(curb_band_event),
            "close_pass_event": float(close_pass_event),
        }

    def transitions_and_costs(self, st: E4BikeState, action: Tuple[Any, int, Any, int]) -> Tuple[Dict[E4BikeState, float], np.ndarray]:
        if self.is_terminal(st):
            return {}, np.zeros(4, dtype=float)
        cache_key = (st, action)
        if self.cache_edges and cache_key in self._edge_cache:
            trans, c = self._edge_cache[cache_key]
            return trans, c.copy()

        CENTER = int(self.e4.CENTER)
        FT_3_4 = int(self.e4.FT_3_4)
        FT_2_3 = int(self.e4.FT_2_3)
        LT_2FT = int(self.e4.LT_2FT)
        (a_cmd_speed, a_next_lane), b_cmd_speed, (c_cmd_speed, c_next_lane), d_cmd_speed = action
        costs = np.zeros(4, dtype=float)

        if not st.a_done:
            costs[0] += self.costs.A_time
            costs[0] += self.costs.A_speed_effort * max(0, int(a_cmd_speed))
            if self.speed_event_mode == "commanded" and int(a_cmd_speed) > int(self.a_speed_limit):
                costs[0] += self.costs.A_speeding
            if int(a_next_lane) != CENTER:
                costs[0] += {FT_3_4: 1.0, FT_2_3: 2.0, LT_2FT: 5.0}.get(int(a_next_lane), 1.0) * self.costs.A_curb

        if not st.b_done:
            costs[1] += self.costs.B_time
            costs[1] += self.costs.B_speed_effort * max(0, int(b_cmd_speed))
            if self.speed_event_mode == "commanded" and int(b_cmd_speed) > int(self.b_speed_limit):
                costs[1] += self.costs.B_speeding

        if not st.c_done:
            costs[2] += self.costs.C_time
            costs[2] += self.costs.C_speed_effort * max(0, int(c_cmd_speed))
            if self.speed_event_mode == "commanded" and int(c_cmd_speed) > int(self.c_speed_limit):
                costs[2] += self.costs.C_speeding
            if int(c_next_lane) != CENTER:
                costs[2] += {FT_3_4: 1.0, FT_2_3: 2.0, LT_2FT: 5.0}.get(int(c_next_lane), 1.0) * self.costs.C_curb

        if not st.d_done:
            costs[3] += float(getattr(self.costs, "D_time", 0.05))
            costs[3] += float(getattr(self.costs, "D_speed_effort", 0.01)) * max(0, int(d_cmd_speed))

        out: Dict[E4BikeState, float] = {}
        for a_speed, pa in self._actual_speed_dist(a_cmd_speed).items():
            for b_speed, pb in self._actual_speed_dist(b_cmd_speed).items():
                for c_speed, pc in self._actual_speed_dist(c_cmd_speed).items():
                    for d_speed, pd_ in self._actual_speed_dist(d_cmd_speed).items():
                        p = float(pa) * float(pb) * float(pc) * float(pd_)
                        ns, meta = self._one_successor(st, action, a_speed, b_speed, c_speed, d_speed)
                        out[ns] = out.get(ns, 0.0) + p

                        # In realized-speed mode the stage speeding cost must be
                        # averaged over stochastic successor branches, matching
                        # the PCTL speed monitor exactly.
                        if self.speed_event_mode == "realized":
                            if meta["a_speed_event"] > 0:
                                costs[0] += p * self.costs.A_speeding
                            if meta["b_speed_event"] > 0:
                                costs[1] += p * self.costs.B_speeding
                            if meta["c_speed_event"] > 0:
                                costs[2] += p * self.costs.C_speeding

                        if meta["tailgate_event"] > 0:
                            costs[0] += p * self.costs.A_tailgated
                            costs[1] += p * self.costs.B_tailgate
                        if meta["collision_event"] > 0:
                            costs[0] += p * self.costs.A_collision
                            costs[1] += p * self.costs.B_collision
                            costs[2] += p * self.costs.C_collision
                            costs[3] += p * self.costs.D_vehicle_collision_externality
                        if meta["cyclist_collision_event"] > 0:
                            costs[0] += p * self.costs.A_cyclist_collision
                            costs[2] += p * self.costs.C_cyclist_collision
                            costs[3] += p * self.costs.D_collision
                        if meta["curb_band_event"] > 0:
                            costs[3] += p * self.costs.D_curb_band
                        if meta["close_pass_event"] > 0:
                            costs[3] += p * self.costs.D_close_pass

                        if self.is_terminal(ns):
                            if not ns.a_done:
                                costs[0] += p * self.costs.A_terminal_late
                            # B_terminal_late is a late-merge penalty.  Once Bob has
                            # merged into the safe lane, we do not penalize him for
                            # not reaching the road target by the horizon.
                            if not self.bob_is_merged(ns):
                                costs[1] += p * self.costs.B_terminal_late
                            if not ns.c_done:
                                costs[2] += p * self.costs.C_terminal_late
                            if not ns.d_done:
                                costs[3] += p * float(getattr(self.costs, "D_terminal_late", 0.0))

        total = sum(out.values())
        if total > 0 and abs(total - 1.0) > 1e-12:
            for key in list(out):
                out[key] /= total
        if self.cache_edges:
            self._edge_cache[cache_key] = (out, costs.copy())
        return out, costs.copy()

    def transitions(self, st: E4BikeState, action: Tuple[Any, int, Any, int]) -> Dict[E4BikeState, float]:
        return self.transitions_and_costs(st, action)[0]

    def expected_costs4(self, st: E4BikeState, action: Tuple[Any, int, Any, int]) -> np.ndarray:
        return self.transitions_and_costs(st, action)[1]

    def property_value(self, st: E4BikeState, name: str) -> float:
        if name == "a_on_time":
            return float(st.a_done)
        if name == "b_on_time":
            return float(self.bob_is_merged(st))
        if name == "b_merged":
            return float(self.bob_is_merged(st))
        if name == "b_reached_target":
            return float(st.b_done)
        if name == "c_on_time":
            return float(st.c_done)
        if name == "d_on_time":
            return float(st.d_done)
        if name == "a_no_speeding":
            return float(not st.a_speed_bad)
        if name == "b_no_speeding":
            return float(not st.b_speed_bad)
        if name == "c_no_speeding":
            return float(not st.c_speed_bad)
        if name == "no_collision":
            return float(not st.collision_bad)
        if name == "no_cyclist_collision":
            return float(not st.cyclist_collision_bad)
        if name == "tailgate_ok":
            return float(st.tailgate_count <= self.tailgate_limit)
        if name == "state_definition_relaxation_3to4ft":
            return float(st.curb_band_count > 0)
        if name == "curb_band_use":
            return float(st.curb_band_count > 0)
        if name == "strict_4ft_close_pass_use":
            return float(st.curb_band_count > 0 or st.close_pass_count > 0 or st.cyclist_collision_bad)
        if name == "relaxed_3ft_close_pass_use":
            return float(st.close_pass_count > 0 or st.cyclist_collision_bad)
        if name == "close_pass_ok":
            return float(st.close_pass_count <= self.close_pass_limit and not st.cyclist_collision_bad)
        if name == "safe_by_4ft_definition":
            return float(st.curb_band_count == 0 and st.close_pass_count == 0 and not st.cyclist_collision_bad)
        if name == "safe_by_3ft_definition":
            return float(st.close_pass_count == 0 and not st.cyclist_collision_bad)
        if name == "safe_by_2ft_definition":
            return float(not st.cyclist_collision_bad)
        if name == "courtesy_norm":
            ok = (
                (not st.collision_bad)
                and (not st.cyclist_collision_bad)
                and st.tailgate_count <= self.tailgate_limit
            )
            return float(ok)
        raise KeyError(f"Unknown E4MovingBikeWorld property {name!r}")

    def reachable_states(self, *, max_states: int = 1_000_000, verbose: bool = False) -> List[E4BikeState]:
        seen = {self.start_state}
        frontier = [self.start_state]
        while frontier:
            st = frontier.pop()
            if self.is_terminal(st):
                continue
            for action in self.actions_from(st):
                for ns in self.transitions(st, action):
                    if ns not in seen:
                        seen.add(ns)
                        frontier.append(ns)
                        if len(seen) > int(max_states):
                            raise RuntimeError(f"reachable state count exceeded max_states={max_states}")
            if verbose and len(seen) % 50_000 == 0:
                print(f"[reachable_states moving bike] discovered {len(seen):,} states")
        return sorted(seen)

    def format_action(self, action: Tuple[Any, int, Any, int]) -> str:
        (sa, la), sb, (sc, lc), sd = action
        return (
            f"A(speed={sa}, lane={self.e4.LANE_NAMES.get(la, la)}), "
            f"B(speed={sb}, lane=auto-merge), "
            f"C(speed={sc}, lane={self.e4.LANE_NAMES.get(lc, lc)}), "
            f"D_bike(speed={sd})"
        )

    def ascii_map(self, st: Optional[E4BikeState] = None) -> str:
        if st is None:
            st = self.start_state
        rows = {
            int(self.e4.CYCLIST_EDGE): ["·"] * self.road_len,
            int(self.e4.LT_2FT): ["·"] * self.road_len,
            int(self.e4.FT_2_3): ["·"] * self.road_len,
            int(self.e4.FT_3_4): ["·"] * self.road_len,
            int(self.e4.SAFE_LANE): ["·"] * self.road_len,
            self.merge_lane: ["·"] * self.road_len,
        }
        for lane in rows:
            rows[lane][self.target_pos] = "T"
        for x in self.contested_positions_at(st.d_pos):
            rows[int(self.e4.LT_2FT)][x] = "x"
            rows[int(self.e4.FT_2_3)][x] = "y"
            rows[int(self.e4.FT_3_4)][x] = "d"
        if not st.d_done:
            rows[int(self.e4.CYCLIST_EDGE)][int(st.d_pos)] = "D"

        def place(lane: int, pos: int, symbol: str) -> None:
            lane = int(lane) if int(lane) in rows else int(self.e4.SAFE_LANE)
            current = rows[lane][int(pos)]
            rows[lane][int(pos)] = symbol if current == "·" else current + "/" + symbol

        if not st.a_done:
            place(st.a_lane, st.a_pos, "A")
        if not st.b_done:
            place(st.b_lane, st.b_pos, "B")
        if not st.c_done:
            place(st.c_lane, st.c_pos, "C")

        return "\n".join([
            "cyclist/curb: " + " ".join(rows[int(self.e4.CYCLIST_EDGE)]),
            "<2ft hard:    " + " ".join(rows[int(self.e4.LT_2FT)]),
            "2-3ft close:  " + " ".join(rows[int(self.e4.FT_2_3)]),
            "3-4ft defn:   " + " ".join(rows[int(self.e4.FT_3_4)]),
            ">=4ft safe:   " + " ".join(rows[int(self.e4.SAFE_LANE)]),
            "merge lane:    " + " ".join(rows[self.merge_lane]),
            (
                f"t={st.t}, D_pos={st.d_pos}, D_done={st.d_done}, "
                f"B_lane={self.e4.LANE_NAMES.get(st.b_lane, st.b_lane)}, "
                f"B_merged={self.bob_is_merged(st)}, B_done={st.b_done}, "
                f"semantic_3to4ft={st.curb_band_count}, lt3ft={st.close_pass_count}, "
                f"cyclist_collision={st.cyclist_collision_bad}"
            ),
        ])


def make_e4_moving_bike_env(**kwargs) -> E4MovingBikeWorld:
    """Build the local model using this module's constants and cost class."""
    import sys

    return E4MovingBikeWorld(sys.modules[__name__], **kwargs)

