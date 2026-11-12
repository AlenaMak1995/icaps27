import agent as moving_bike
from bargaining_operator import (
    build_two_fleet_bargaining_problem,
    make_default_two_fleet_world,
)
from evaluator import RoadEvaluator
from road_demo import run_demo

HORIZON = 2
DEMO_OPTIONS = {
    "clearance_ft": 4,
    "cost_caps": (60.0, 35.0),
    "tight_cost_caps": (50.0, 27.0),
    "speeding_probability": 0.5,
}

world = make_default_two_fleet_world(moving_bike, horizon=HORIZON)
problem = build_two_fleet_bargaining_problem(world)
evaluator = RoadEvaluator(problem)
solve_o1 = evaluator.solve_o1
solve_o2 = evaluator.solve_o2
evaluate_pair = evaluator.evaluate_pair

if __name__ == "__main__":
    print(f"{len(problem.lp_data.nonterminal_states)} decision states, "
          f"{problem.lp_data.E} joint state-action edges")
    run_demo(evaluator, **DEMO_OPTIONS)
