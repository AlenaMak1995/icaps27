"""Narrate the two-fleet road evaluator examples."""

import json
from collections import defaultdict

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


def _report_response(operator, response, cap):
    speaker = f"Operator {operator}"
    status = response["status"]
    if status == "infeasible":
        print(
            f"{speaker} | With that policy fixed, I have no response that meets "
            "the norm and safety rules."
        )
        return

    cost = response["minimum_cost"]
    if status == "cost_cap_exceeded":
        print(f"{speaker} | My cheapest response costs {cost:g}, which is above my cap of {cap:g}.")
    else:
        print(f"{speaker} | My cheapest response costs {cost:g}, within my cap of {cap:g}.")


def _report_pair(label, pair):
    checks = pair["checks"]
    if pair["status"] == "feasible":
        print(f"Evaluator | {label} meets the norm, safety rules, and both cost caps.")
    else:
        print(f"Evaluator | {label} is infeasible.")
        if any(not row["satisfied"] for row in checks["norm"].values()):
            print("  The pair fails a norm requirement.")
        if any(not row["satisfied"] for row in checks["hard"].values()):
            print("  The pair fails a safety requirement.")

    for fleet in (1, 2):
        cap_check = checks["cost_caps"][f"fleet{fleet}"]
        comparison = "within" if cap_check["satisfied"] else "above"
        print(
            f"  Operator {fleet} costs {cap_check['cost']:g}, "
            f"{comparison} the cap of {cap_check['cap']:g}."
        )


def _run_speeding_example(evaluator, reference_2, norm, clearance_ft, caps, probability):
    print("\nComparing two no-speeding thresholds")
    if probability == 0:
        print("Evaluator | Set a positive speeding probability to run this comparison.")
        return

    data = evaluator.problem.lp_data
    start = data.state_index[data.env.start_state]
    start_actions = evaluator.actions[1][start]
    reference_action = reference_2[start]
    lead, merging, (reference_speed, lane), cyclist = start_actions[reference_action]
    fleet_2 = data.env.fleet2
    speed_limit = fleet_2.c_speed_limit
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
        print(
            "Evaluator | This example needs a legal-speed command and a matching "
            "speeding command at the start."
        )
        return

    mixed_fleet_2 = dict(reference_2)
    mixed_fleet_2[start] = {reference_action: 1 - probability, speeding_action: probability}
    print(
        "Operator 2 | I propose using a speeding command on the first step "
        f"with probability {100 * probability:g}%."
    )
    print("Operator 1 | I'll fix that policy and solve O1 under the initial 100% no-speeding rule.")
    nominal_response = evaluator.solve_o1(norm, mixed_fleet_2, caps[0])
    relaxed_threshold = 1 - probability
    if nominal_response["status"] == "infeasible" and fleet_2.speed_event_mode == "commanded":
        print(
            "Operator 1 | My response is infeasible. Your first-step command makes your "
            f"chance of avoiding speeding at most {100 * relaxed_threshold:g}%, below the required 100%."
        )
    else:
        _report_response(1, nominal_response, caps[0])

    print(
        "Operator 2 | I propose lowering the no-speeding threshold "
        f"to {100 * relaxed_threshold:g}% for both fleets."
    )
    print("Operator 1 | I'll solve O1 again with your same policy under that threshold.")
    relaxed_norm = road_norm(relaxed_threshold, clearance_ft)
    relaxed_response = evaluator.solve_o1(relaxed_norm, mixed_fleet_2, caps[0])
    _report_response(1, relaxed_response, caps[0])
    if relaxed_response["status"] == "optimal":
        # JSON converts the integer state and action IDs to string keys.
        policy_1 = json.loads(json.dumps(relaxed_response["policy"]))
        policy_2 = json.loads(json.dumps(mixed_fleet_2))
        relaxed_pair = evaluator.evaluate_pair(relaxed_norm, policy_1, policy_2, *caps)
        _report_pair("The pair under the counteroffer", relaxed_pair)
        speed_check = relaxed_pair["checks"]["norm"]["op2_tau"]
        print(
            f"  Operator 2 avoids speeding in {100 * speed_check['probability']:g}% of episodes. "
            f"The required minimum is {100 * speed_check['minimum']:g}%."
        )


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

    horizon = problem.lp_data.env.horizon
    print("\nScripted road bargaining walkthrough")
    print(
        f"Setup | For this {horizon}-step run, the initial norm requires {clearance_ft:g} ft "
        "of clearance and a 100% chance of a speeding-free episode for each ER."
    )
    norm = road_norm(1.0, clearance_ft)
    try:
        reference_1, reference_2 = make_reference_policies(problem, evaluator, norm)
    except ReferenceInfeasibleError:
        print(
            f"Evaluator | The joint reference LP is infeasible under this norm "
            f"at horizon {horizon}."
        )
        return

    reference_pair = evaluator.evaluate_pair(norm, reference_1, reference_2, alpha_1, alpha_2)
    print("Setup | A joint optimization supplied one reference policy for each operator.")
    _report_pair("The reference pair", reference_pair)

    print("\nResponses to the reference policies")
    print(f"Operator 1 | My cap is {alpha_1:g}. I'll fix Operator 2's reference policy and solve O1.")
    response_1 = evaluator.solve_o1(norm, reference_2, alpha_1)
    _report_response(1, response_1, alpha_1)
    print(f"Operator 2 | My cap is {alpha_2:g}. I'll fix Operator 1's reference policy and solve O2.")
    response_2 = evaluator.solve_o2(norm, reference_1, alpha_2)
    _report_response(2, response_2, alpha_2)

    print("\nChecking alternative cost caps against the reference policies")
    print(f"Operator 1 | I'll check O1 with a cost cap of {tight_alpha_1:g}.")
    tight_o1 = evaluator.solve_o1(norm, reference_2, tight_alpha_1)
    _report_response(1, tight_o1, tight_alpha_1)
    print(f"Operator 2 | I'll check O2 with a cost cap of {tight_alpha_2:g}.")
    tight_o2 = evaluator.solve_o2(norm, reference_1, tight_alpha_2)
    _report_response(2, tight_o2, tight_alpha_2)
    tight_pair = evaluator.evaluate_pair(norm, reference_1, reference_2, tight_alpha_1, alpha_2)
    _report_pair("The reference pair with Operator 1's comparison cap", tight_pair)

    print("\nChecking the two returned policies together")
    if response_1["status"] == response_2["status"] == "optimal":
        actual_pair = evaluator.evaluate_pair(
            norm, response_1["policy"], response_2["policy"], alpha_1, alpha_2
        )
        _report_pair("The pair made from the original O1 and O2 responses", actual_pair)
    else:
        print("Evaluator | I need a returned policy from each operator to check the pair.")

    _run_speeding_example(
        evaluator, reference_2, norm, clearance_ft, (alpha_1, alpha_2), speeding_probability
    )
