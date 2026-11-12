import agent as moving_bike
from bargaining_operator import (
    build_two_fleet_bargaining_problem,
    make_default_two_fleet_world,
)
from evaluator import (
    RoadEvaluator
)

world = make_default_two_fleet_world(moving_bike, horizon=2)
problem = build_two_fleet_bargaining_problem(world)
print(f"{len(problem.lp_data.nonterminal_states)} decision states, "
      f"{problem.lp_data.E} joint state-action edges")


evaluator = RoadEvaluator(problem)
solve_o1 = evaluator.solve_o1
solve_o2 = evaluator.solve_o2
evaluate_pair = evaluator.evaluate_pair      