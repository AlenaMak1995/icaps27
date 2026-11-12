# icaps27
## About this demo

This is a small starting example for getting familiar with the road model and testing bargaining proposals. The setup can grow as needed. We can let the cyclists move, give vehicles more lane choices, enable additional passing-distance bands, lengthen the road, or give each episode more time steps. These changes can increase the number of road states and combinations of fleet actions the evaluator has to consider. Starting small makes it easier to see why the norm passes or fails before trying a larger scenario.

## Road scenario and rules

Two fleets share a road with 12 longitudinal cells. Each fleet has four road users: Alice (the lead car), Bob (the merging car), an emergency vehicle (ER), and a cyclist. Each fleet is one bargaining operator. At each of the time steps, its policy chooses actions for its four road users based on the shared road state.

![Starting positions of both fleets on the shared 12-cell road](road_layout.png)

| Rule | How the evaluator checks it |
|---|---|
| **Collision safety** | The probability of completing the episode without a vehicle collision or a cyclist collision must be 1. This applies within and across fleets. |
| **ER speed** | Each fleet’s ER has a speed limit of 2 cells per step. The evaluator checks the probability that each ER stays within that limit for the entire episode. The initial norm requires satisfaction of the speed limit with probability 1. |
| **Cyclist clearance** | The cyclists occupy a bike lane along the edge of the road. Under the initial rule, a vehicle from either fleet must leave at least 4 ft of space when passing a cyclist. During negotiation, either fleet can propose lowering that minimum to 3 ft or 2 ft. For example, a pass at 3.5 ft would fail the initial rule and pass under a 3 ft agreement. |

## What each road user can do and what it costs

All costs are in **model units**. Lower cost is better. A lane-use cost is charged to the vehicle choosing that lane. A close-pass cost is charged to the cyclist who experiences the encounter.

| Road user | Actions in this demo | Main costs |
|---|---|---|
| **Alice — lead car** | Moves 1 cell per step. In eligible states, she can choose the 4 ft lane or the 3–4 ft band. | `0.16` per active step, `+2.50` for choosing the 3–4 ft band, and `+8` if she has yet to reach the road end when the episode ends. The model also charges her `+4` when Bob tailgates her. |
| **Bob — merging car** | Moves 1 cell per step. The model moves him from the merge lane into the 4 ft lane when he reaches his merge point. | `2.43` per active step and `+32` if he has yet to merge when the episode ends. The model also charges him `+1.20` for tailgating Alice. |
| **ER — emergency vehicle** | Chooses speed 2 or 3. In eligible states, it can choose the center lane or the 3–4 ft band. | `4.54` per active step at speed 2 or `4.56` at speed 3, plus `0.05` for speeding at speed 3. Choosing the 3–4 ft band adds `0.20`. Ending the episode before reaching the road end adds `70`. |
| **Cyclist** | Stays at its starting cell along the road edge. | `0.05` per active step and `+2` for a vehicle encounter in the 3–4 ft band. The model also defines a closer-pass cost of `8` and a cyclist-collision cost of `500`. |

The model assigns collision penalties when scoring a policy pair, and the hard safety rule makes any pair with positive collision probability infeasible. The two-step horizon ends before either fleet’s Alice or ER can reach the road end and before either Bob can merge, so those terminal charges are fixed in this demo.

**Example.** Fleet 2’s Alice starts at cell 7. If she moves to cell 8 in the 3–4 ft band, she pays `2.50` for using that band, and Fleet 1’s cyclist at cell 9 incurs `2` for the encounter. That encounter violates the 4 ft clearance rule and satisfies the 3 ft definition. One fleet’s action can therefore affect the other fleet’s cost and whether the pair passes the proposed norm.

## Tailgating and road norms

**Tailgating** means that Bob ends a step **one cell behind Alice in the same lane**, while both are still driving. A gap of **two or more cells** is normal following distance for this check. The model charges Bob `1.20` for tailgating and Alice `4.00` for being tailgated. 

| Rule | Initial setting | What a proposal can change |
|---|---|---|
| **Collision safety** | There must be a **100% probability** of avoiding vehicle and cyclist collisions throughout the episode, including between fleets. | Fixed. Every proposal must satisfy this requirement. |
| **ER speed** | Each fleet’s ER must stay at or below **2 cells per step** for the whole episode with **100% probability**. | A new norm can lower the required probability. For example, a **50%** requirement means that at least half of the possible runs must have zero speeding by that ER. |
| **Cyclist clearance** | Each cyclist must receive at least **4 ft of clearance** on every modeled pass, with **100% probability**. | A new norm can lower the minimum clearance to **3 ft** or **2 ft**. The closest available passing band in this demo is **3–4 ft**, so both relaxed definitions allow the same passes here. |

The **initial negotiable norm** combines the 100% ER speed requirement and the 4 ft clearance rule. Collision safety is a hard requirement for every agreement.

## How fleet costs are calculated

Each fleet’s cost is a weighted sum of its four road users’ expected costs. The same road event can matter differently to the two operators because they use different weights.

| Operator | Alice | Bob | ER | Cyclist | Demo cost cap |
|---|---:|---:|---:|---:|---:|
| Fleet 1 | 0.10 | 0.10 | 0.60 | 0.20 | 60 |
| Fleet 2 | 0.10 | 0.10 | 0.30 | 0.50 | 35 |

## Model files

| File | What it does |
|---|---|
| `agent.py` | Defines the four road users, their state and actions, movement rules, event flags, and per-user cost coefficients in `E4CostParams`. Uses NumPy. |
| `bargaining_operator.py` | Places two fleets on the same road, adds interactions between them, and builds the joint model with fleet costs and norm checks. Uses the first module, NumPy, and SciPy. |
