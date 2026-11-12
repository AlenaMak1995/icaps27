from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import spsolve


def road_norm(no_speeding_probability, clearance_ft):
    """Convert a road offer to (tau, sigma_4ft, sigma_3ft, sigma_2ft)."""
    profiles = {4: (0.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0), 2: (1.0, 1.0, 0.0)}
    probability = float(no_speeding_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("no_speeding_probability must be in [0, 1].")
    if clearance_ft not in profiles:
        raise ValueError("clearance_ft must be 4, 3, or 2.")
    return (1.0 - probability, *profiles[clearance_ft])


class RoadEvaluator:
    """Conditional policy optimization on the supplement's two-fleet model."""

    def __init__(self, problem):
        self.problem = problem
        self.data = problem.lp_data
        self.tolerance = problem.numerical_tolerance
        self.state_count = len(self.data.nonterminal_states)
        self.edge_ids = np.arange(self.data.E)
        self.edge_states = np.array([self.data.state_index[s] for s, _ in self.data.edges])
        self.norm_matrix = csr_matrix(problem.relaxation.G_relax)
        self.hard_names = tuple(problem.hard_thresholds)
        self.hard_minima = np.array([problem.hard_thresholds[name] for name in self.hard_names])
        self.hard_matrix = csr_matrix(np.vstack([self.data.q[name] for name in self.hard_names]))
        self.constraint_matrix = vstack([self.norm_matrix, -self.hard_matrix], format="csr")

        action_ids = [[{} for _ in range(self.state_count)] for _ in range(2)]
        self.edge_actions = np.empty((2, self.data.E), dtype=int)
        for edge, ((_, actions), state_id) in enumerate(zip(self.data.edges, self.edge_states)):
            for fleet, action in enumerate(actions):
                ids = action_ids[fleet][state_id]
                self.edge_actions[fleet, edge] = ids.setdefault(action, len(ids))
        self.actions = tuple(tuple(tuple(ids) for ids in fleet) for fleet in action_ids)
        self.action_offsets = tuple(
            np.cumsum([0, *(len(actions) for actions in fleet)]) for fleet in self.actions
        )

    @staticmethod
    def _index(value, label):
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        raise ValueError(f"{label} must be an integer ID.")

    def _policy(self, fleet, policy):
        if not isinstance(policy, Mapping):
            raise ValueError("A policy must map every nonterminal state ID to an action or distribution.")
        result = {}
        for state, choice in policy.items():
            state = self._index(state, "State")
            if state in result or not 0 <= state < self.state_count:
                raise ValueError(f"Invalid or repeated state ID {state}.")
            if not isinstance(choice, Mapping):
                choice = {choice: 1.0}
            probabilities = {}
            for action, probability in choice.items():
                action = self._index(action, "Action")
                if action in probabilities or not 0 <= action < len(self.actions[fleet][state]):
                    raise ValueError(f"Invalid or repeated action ID {action} at state {state}.")
                probability = float(probability)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"Invalid action probability at state {state}.")
                probabilities[action] = probability
            total = sum(probabilities.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"Action probabilities must sum to one at state {state}.")
            result[state] = {action: value / total for action, value in probabilities.items() if value > 0.0}
        missing = sorted(set(range(self.state_count)) - result.keys())
        if missing:
            raise ValueError(f"Policy is missing nonterminal state IDs {missing}.")
        return result

    def _norm_rhs(self, norm):
        z = np.asarray(norm, dtype=float)
        caps = self.problem.z_max
        if z.shape != (4,) or not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z > caps):
            raise ValueError("Use four norm slacks within the model's bounds.")
        profiles = ([0, 0, 0], [caps[1], 0, 0], [caps[1], caps[2], 0])
        if not any(np.array_equal(z[1:], profile) for profile in profiles):
            raise ValueError("Use an ordered 4 ft, 3 ft, or 2 ft clearance profile.")
        return self.problem.relaxation.slack_map @ z - self.problem.relaxation.b_relax

    @staticmethod
    def _cap(cap):
        cap = float(cap)
        if not np.isfinite(cap):
            raise ValueError("A cost cap must be finite.")
        return cap

    def _edge_probabilities(self, fleet, policy):
        return np.array([
            policy[state].get(action, 0.0)
            for state, action in zip(self.edge_states, self.edge_actions[fleet])
        ])

    def _pair_occupancy(self, policy1, policy2):
        weights = self._edge_probabilities(0, policy1) * self._edge_probabilities(1, policy2)
        pair_map = csr_matrix(
            (weights, (self.edge_ids, self.edge_states)), shape=(self.data.E, self.state_count)
        )
        flow = self.data.A_flow @ pair_map
        state_mass = spsolve(flow.tocsc(), self.data.b_flow)
        if not np.all(np.isfinite(state_mass)) or np.min(state_mass) < -self.tolerance:
            raise RuntimeError("The submitted policies did not yield a valid state flow.")
        occupancy = np.asarray(pair_map @ np.maximum(state_mass, 0.0))
        residual = self.data.A_flow @ occupancy - self.data.b_flow
        if np.max(np.abs(residual)) > self.tolerance:
            raise RuntimeError("The submitted policies failed the occupation-flow check.")
        return occupancy

    def _costs(self, occupancy):
        return dict(zip(("fleet1", "fleet2"), map(float, self.data.C @ occupancy)))

    def _minimum_check(self, probability, minimum):
        return {
            "probability": float(probability),
            "minimum": float(minimum),
            "satisfied": bool(probability >= minimum - self.tolerance),
        }

    def _cost_check(self, cost, cap):
        return {"cost": cost, "cap": cap, "satisfied": bool(cost <= cap + self.tolerance)}

    def _constraint_checks(self, occupancy, norm_rhs):
        return {
            "norm": {
                name: self._minimum_check(probability, minimum)
                for name, probability, minimum in zip(
                    self.problem.relaxation.row_names, -(self.norm_matrix @ occupancy), -norm_rhs
                )
            },
            "hard": {
                name: self._minimum_check(probability, minimum)
                for name, probability, minimum in zip(
                    self.hard_names, self.hard_matrix @ occupancy, self.hard_minima
                )
            },
        }

    def evaluate_pair(self, norm, x1, x2, alpha1, alpha2):
        """Check the same submitted pair against shared constraints and both caps."""
        norm_rhs = self._norm_rhs(norm)
        caps = (self._cap(alpha1), self._cap(alpha2))
        occupancy = self._pair_occupancy(self._policy(0, x1), self._policy(1, x2))
        costs = self._costs(occupancy)
        checks = self._constraint_checks(occupancy, norm_rhs)
        checks["cost_caps"] = {
            name: self._cost_check(cost, cap) for (name, cost), cap in zip(costs.items(), caps)
        }
        feasible = all(row["satisfied"] for group in checks.values() for row in group.values())
        return {"status": "feasible" if feasible else "infeasible", "costs": costs, "checks": checks}

    def _best_response(self, fleet, norm, other_policy, cap):
        norm_rhs = self._norm_rhs(norm)
        cap = self._cap(cap)
        fixed = self._policy(1 - fleet, other_policy)
        offsets = self.action_offsets[fleet]
        own_action_columns = offsets[self.edge_states] + self.edge_actions[fleet]
        response_map = csr_matrix(
            (self._edge_probabilities(1 - fleet, fixed), (self.edge_ids, own_action_columns)),
            shape=(self.data.E, int(offsets[-1])),
        )
        objective = np.asarray(self.data.C[fleet] @ response_map).ravel()
        result = linprog(
            objective,
            A_ub=self.constraint_matrix @ response_map,
            b_ub=np.concatenate((norm_rhs, -self.hard_minima)),
            A_eq=self.data.A_flow @ response_map,
            b_eq=self.data.b_flow,
            bounds=(0, None),
            method="highs",
        )
        # The cap bounds the objective itself, so a minimum above the cap proves infeasibility.
        if result.status == 2:
            return {"status": "infeasible", "reason": "hard_or_norm_constraints",
                    "minimum_cost": None, "policy": None, "checks": None}
        if not result.success:
            raise RuntimeError(f"Best-response LP failed: {result.message}")

        policy = {}
        for state, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            masses = np.maximum(result.x[start:stop], 0.0)
            mass = masses.sum()
            policy[state] = (
                {action: float(weight / mass) for action, weight in enumerate(masses) if weight > 0.0}
                if mass > 0.0 else {0: 1.0}
            )
        occupancy = self._pair_occupancy(policy, fixed) if fleet == 0 else self._pair_occupancy(fixed, policy)
        minimum_cost = float(self.data.C[fleet] @ occupancy)
        checks = self._constraint_checks(occupancy, norm_rhs)
        if (abs(minimum_cost - result.fun) > self.tolerance
                or not all(row["satisfied"] for group in checks.values() for row in group.values())):
            raise RuntimeError("Recovered policy failed to reproduce the best-response LP solution.")
        cap_check = self._cost_check(minimum_cost, cap)
        checks["cost_caps"] = {f"fleet{fleet + 1}": cap_check}
        return {
            "status": "optimal" if cap_check["satisfied"] else "cost_cap_exceeded",
            "minimum_cost": minimum_cost,
            "policy": policy if cap_check["satisfied"] else None,
            "checks": checks,
        }

    def solve_o1(self, norm, x2_hat, alpha1):
        """Minimize Fleet 1's expected cost under a fixed Fleet 2 policy."""
        return self._best_response(0, norm, x2_hat, alpha1)

    def solve_o2(self, norm, x1_hat, alpha2):
        """Minimize Fleet 2's expected cost under a fixed Fleet 1 policy."""
        return self._best_response(1, norm, x1_hat, alpha2)