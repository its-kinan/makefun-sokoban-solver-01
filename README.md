# Sokoban Solver — push-optimal, room by room

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Sokoban solver](https://img.shields.io/badge/sokoban-push--optimal-green)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Sokoban solver for make.fun's **"Build the best Sokoban solver, 21 November 2026"** bounty ($100 in USDC).
It answers every instance of a set in one run, in as few pushes as the level allows.

## How it works

The levels are a hallway with independent rooms off it: a box pushed out of a
room can never come back, so each room is a problem of its own and the level's
push-optimum is the sum of its rooms' optima. The solver

1. **cuts the hallway out of the floor** with the dead-square test (a box on a
   square from which no pull-back from a goal reaches can never be solved),
   splitting the board into rooms;
2. **solves each room push-optimally** with A\* over pushes — a state is the
   boxes plus the player's reachable region — guided by a Hungarian-assignment
   heuristic (min total box-to-goal distance) and pruned by dead squares and
   frozen 2×2 blocks;
3. **stitches the rooms together** with plain player walks through the hallway.

A\* over pushes with an admissible heuristic is push-optimal per room, so the
whole level is push-optimal: full marks on every instance it solves.

## Running it

```sh
node main.ts            # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` override the paths (as in the kit's
starter). Node 22, no dependencies, no network. Deterministic: budgets are
counted in expanded states, never wall-clock, so two runs write the same bytes.

## Results

Practice set: **1,000,000 / 1,000,000** — every instance solved in exactly its
push-optimal count (13, 16, 29, 47, 60 pushes), whole set in under a second.
100 further freshly drawn sets (500 instances) all solved at their optima too.

## License

MIT — see [LICENSE](LICENSE).
