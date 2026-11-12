"""Run the two-fleet road evaluator examples."""

import json
from collections import defaultdict
from pprint import pprint

import numpy as np
from scipy.optimize import linprog

from evaluator import road_norm


class ReferenceInfeasibleError(RuntimeError):
    """The centralized reference LP has no feasible solution."""


def make_reference_policies(problem, evaluator, norm):
    """Extract deterministic reference policies from a centralized sum-cost LP."""
    data = problem.lp_data
    relaxation = problem.relaxation
    safety_rows = [-data.q[name] for name in problem.hard_thresholds]
    safety_rhs = -np.array(list(problem.hard_thresholds.values()))
    reference = linprog(
        data.C.sum(axis=0),
        A_ub=np.vstack([relaxation.G_relax, *safety_rows]),
        b_ub=np.concatenate([
            relaxation.slack_map @ np.asarray(norm) - relaxation.b_relax,
            safety_rhs,
        ]),
        A_eq=data.A_flow,
        b_eq=data.b_flow,
        bounds=(0, None),
        method="highs",
    )
    if reference.status == 2:
        raise ReferenceInfeasibleError(reference.message)
    if not reference.success:
        raise RuntimeError(f"Could not build reference policies: {reference.message}")

    edges_by_state = defaultdict(list)
    for edge_id, (state, joint_action) in enumerate(data.edges):
        edges_by_state[state].append((reference.x[edge_id], joint_action))

    policies = ({}, {})
    for state_id, state in enumerate(data.nonterminal_states):
        # Both fleets use the components of the same joint action. Ties at
        # states with zero occupancy use the first available joint action.
        _, joint_action = max(edges_by_state[state], key=lambda edge: edge[0])
        for fleet, policy in enumerate(policies):
            policy[state_id] = evaluator.actions[fleet][state_id].index(
                joint_action[fleet]
            )
    return policies


def _run_speeding_example(evaluator, reference_2, norm, clearance_ft, caps, probability):
    print(f"\nO1 with Fleet 2's first-step speeding command probability {probability:g}")
    if probability == 0:
        print("Choose a positive speeding probability to run this comparison.")
        return

    data = evaluator.problem.lp_data
    start = data.state_index[data.env.start_state]
    start_actions = evaluator.actions[1][start]
    reference_action = reference_2[start]
    lead, merging, (reference_speed, lane), cyclist = start_actions[reference_action]
    speed_limit = data.env.fleet2.c_speed_limit
    speeding_action = next(
        (
            index
            for index, (other_lead, other_merging, (speed, other_lane), other_cyclist)
            in enumerate(start_actions)
            if (other_lead, other_merging, other_lane, other_cyclist)
            == (lead, merging, lane, cyclist)
            and speed > speed_limit
        ),
        None,
    )
    if reference_speed > speed_limit or speeding_action is None:
        print("This comparison needs an allowed, otherwise identical speeding action at the start.")
        return

    mixed_fleet_2 = dict(reference_2)
    mixed_fleet_2[start] = {reference_action: 1 - probability, speeding_action: probability}
    nominal_response = evaluator.solve_o1(norm, mixed_fleet_2, caps[0])
    relaxed_threshold = 1 - probability
    relaxed_norm = road_norm(relaxed_threshold, clearance_ft)
    relaxed_response = evaluator.solve_o1(relaxed_norm, mixed_fleet_2, caps[0])
    summary = {
        "nominal_status": nominal_response["status"],
        "relaxed_no_speeding_threshold": relaxed_threshold,
        "relaxed_status": relaxed_response["status"],
    }
    if relaxed_response["status"] == "optimal":
        # JSON converts the integer state and action IDs to string keys.
        policy_1 = json.loads(json.dumps(relaxed_response["policy"]))
        policy_2 = json.loads(json.dumps(mixed_fleet_2))
        relaxed_pair = evaluator.evaluate_pair(relaxed_norm, policy_1, policy_2, *caps)
        summary.update(
            relaxed_pair_status=relaxed_pair["status"],
            relaxed_pair_costs=relaxed_pair["costs"],
            fleet2_no_speeding=relaxed_pair["checks"]["norm"]["op2_tau"],
        )
    pprint(summary)


def run_demo(
    evaluator,
    *,
    clearance_ft=4,
    cost_caps=(60.0, 35.0),
    tight_cost_caps=(50.0, 27.0),
    speeding_probability=0.5,
):
    """Run the reference, conditional-response, and mixed-policy examples."""
    problem = evaluator.problem
    alpha_1, alpha_2 = map(float, cost_caps)
    tight_alpha_1, tight_alpha_2 = map(float, tight_cost_caps)
    if not np.all(np.isfinite([alpha_1, alpha_2, tight_alpha_1, tight_alpha_2])):
        raise ValueError("Cost caps must be finite.")
    speeding_probability = float(speeding_probability)
    if not 0 <= speeding_probability <= 1:
        raise ValueError("speeding_probability must be between 0 and 1.")

    norm = road_norm(1.0, clearance_ft)
    try:
        reference_1, reference_2 = make_reference_policies(problem, evaluator, norm)
    except ReferenceInfeasibleError:
        print(
            f"Reference LP infeasible at horizon {problem.lp_data.env.horizon} "
            f"with {clearance_ft} ft clearance and full speed compliance."
        )
        print("Try a shorter horizon or a different allowed clearance in main.py.")
        return

    reference_pair = evaluator.evaluate_pair(norm, reference_1, reference_2, alpha_1, alpha_2)
    print("Reference pair and costs")
    pprint({"status": reference_pair["status"], "costs": reference_pair["costs"]})

    response_1 = evaluator.solve_o1(norm, reference_2, alpha_1)
    response_2 = evaluator.solve_o2(norm, reference_1, alpha_2)
    print(f"\nConditional responses with caps {alpha_1:g} and {alpha_2:g}")
    for name, response in (("O1", response_1), ("O2", response_2)):
        cost = response["minimum_cost"]
        pprint({
            "problem": name,
            "status": response["status"],
            "minimum_cost": round(cost, 3) if cost is not None else None,
        })

    tight_o1 = evaluator.solve_o1(norm, reference_2, tight_alpha_1)
    tight_o2 = evaluator.solve_o2(norm, reference_1, tight_alpha_2)
    print(f"\nConditional responses with comparison caps {tight_alpha_1:g} and {tight_alpha_2:g}")
    pprint({"O1": tight_o1["status"], "O2": tight_o2["status"]})
    tight_pair = evaluator.evaluate_pair(norm, reference_1, reference_2, tight_alpha_1, alpha_2)
    print(f"\nReference pair with Fleet 1 cap set to {tight_alpha_1:g}")
    pprint({"status": tight_pair["status"], "costs": tight_pair["costs"]})

    print("\nActual pair made from the returned policies")
    if response_1["status"] == response_2["status"] == "optimal":
        actual_pair = evaluator.evaluate_pair(
            norm, response_1["policy"], response_2["policy"], alpha_1, alpha_2
        )
        pprint({"status": actual_pair["status"], "costs": actual_pair["costs"]})
    else:
        print("Both solves must return a policy before the pair can be evaluated.")

    _run_speeding_example(
        evaluator, reference_2, norm, clearance_ft, (alpha_1, alpha_2), speeding_probability
    )
