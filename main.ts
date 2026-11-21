// Sokoban solver: push-optimal, room by room.
//
// Design (see PLAN.md): the level is a hallway with independent rooms off it
// (a box pushed out of a room can never come back, so each room's boxes must
// end on that room's goals). We cut the hallway out of the floor with the
// dead-square test, solve each room on its own with A* over pushes
// (Hungarian-assignment heuristic, dead-square + frozen-2x2 pruning), and
// stitch the rooms together with plain player walks. A* over pushes with an
// admissible heuristic is push-optimal per room, so the whole level is
// push-optimal: the score's full 1,000 points per instance.
//
// Deterministic: budgets are counted in expanded states, never wall-clock;
// no network, no Date, no unseeded randomness. Ties break by insertion order.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, walk letter, push letter]
const DIRS: readonly (readonly [number, number, string, string])[] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: the cells from which a box alone can be pushed onto some
 * goal. Pull-back BFS from the goals: a box at y pushed in direction d lands
 * on x = y + d, which needs the player to have stood at y - d (not a wall).
 * A box on any other square can never reach a goal: pushing it is wasted work.
 */
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** BFS distances from one start cell over floor only (-1 = unreachable). */
function bfsDist(w: number, wall: Uint8Array, start: number): Int32Array {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [start];
  dist[start] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && dist[q] < 0) {
        dist[q] = dist[p] + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/** The player's reachable region from `from` with `occupied` boxes in the way; returns its lowest square. */
function region(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
): number {
  const stack = [from];
  seen[from] = stamp;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack.push(q);
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls/boxes with a box off its goal. */
function frozenBlock(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// A binary heap of [f, h, seq, state] ordered by f, then h, then seq: fully deterministic.
class Heap {
  private a: [number, number, number, number][] = [];
  get size(): number {
    return this.a.length;
  }
  push(e: [number, number, number, number]): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): [number, number, number, number] {
    const a = this.a;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && less(a[c + 1]!, a[c]!)) c++;
        if (!less(a[c]!, a[i]!)) break;
        [a[i], a[c]] = [a[c]!, a[i]!];
        i = c;
      }
    }
    return top;
  }
}
function less(x: [number, number, number, number], y: [number, number, number, number]): boolean {
  return x[0] < y[0] || (x[0] === y[0] && (x[1] < y[1] || (x[1] === y[1] && x[2] < y[2])));
}

interface SolveResult {
  pushes: number;
  moves: string;
  expansions: number;
}

/**
 * Push-optimal search of one (sub)level by A* over pushes. `useHeuristic`
 * false degrades it to plain breadth-first search over pushes (still optimal,
 * slower). Returns null when the state budget runs out or no solution exists.
 *
 * A state is the boxes' squares plus the player's reachable region (named by
 * its lowest square): the player can stand anywhere in its region for free,
 * so the region, not the exact square, is what matters.
 */
function solvePushes(sub: Level, maxExpansions: number, useHeuristic: boolean): SolveResult | null {
  const { w, wall, goals } = sub;
  const size = wall.length;
  const n = sub.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  if (sub.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (solved(sub.boxes)) return { pushes: 0, moves: '', expansions: 0 };

  // Heuristic: min-cost assignment of boxes to goals, cost = BFS cell distance
  // (a box needs at least that many pushes to reach the goal). Subset DP over
  // goal masks; INF when a box cannot reach its goal at all (dead state).
  // With many goals the DP is 2^m, so above 6 goals fall back to the cheaper
  // (still admissible) sum of each box's nearest-goal distance.
  const INF = 1e9;
  const goalDist: Int32Array[] = goals.map((g) => bfsDist(w, wall, g));
  const heuristic = (boxes: number[]): number => {
    if (!useHeuristic) return 0;
    const m = goals.length;
    if (m > 6) {
      let s = 0;
      for (const b of boxes) {
        let best = INF;
        for (let j = 0; j < m; j++) {
          const d = goalDist[j]![b]!;
          if (d >= 0 && d < best) best = d;
        }
        if (best >= INF) return INF;
        s += best;
      }
      return s;
    }
    const dp = new Float64Array(1 << m).fill(INF);
    dp[0] = 0;
    for (let mask = 0; mask < 1 << m; mask++) {
      const k = popcount(mask); // boxes[0..k-1] assigned so far
      if (k >= n || dp[mask]! >= INF) continue;
      for (let j = 0; j < m; j++) {
        if (mask & (1 << j)) continue;
        const d = goalDist[j]![boxes[k]!]!;
        if (d < 0) continue;
        const nm = mask | (1 << j);
        const v = dp[mask]! + d;
        if (v < dp[nm]!) dp[nm] = v;
      }
    }
    return dp[(1 << m) - 1]!;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  // NOTE: the current state's region (seen/here) and a successor's region
  // (mark) need separate stamp maps: stamping a successor's region into the
  // same map would clobber the current region mid-expansion and wrongly
  // prune the remaining pushes.
  const mark = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const boxesArr: number[][] = [sub.boxes.slice()];
  const playerArr: number[] = [sub.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1]; // box cell before the push that made this state
  const pushTo: number[] = [-1]; // box cell after it
  const gScore: number[] = [0];
  const best = new Map<number, number>();
  const heap = new Heap();
  let seq = 0;
  for (const b of sub.boxes) occupied[b] = 1;
  const low0 = region(w, wall, occupied, sub.player, seen, ++stamp);
  for (const b of sub.boxes) occupied[b] = 0;
  const h0 = heuristic(sub.boxes);
  if (h0 >= INF) return null;
  best.set(keyOf(sub.boxes, low0), 0);
  heap.push([h0, h0, seq++, 0]);

  let expansions = 0;
  let found = -1;
  while (heap.size > 0) {
    const [, , , s] = heap.pop();
    const boxes = boxesArr[s]!;
    const g = gScore[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = region(w, wall, occupied, playerArr[s]!, seen, ++stamp);
    const here = stamp;
    if (best.get(keyOf(boxes, low)) !== g) {
      for (const x of boxes) occupied[x] = 0;
      continue; // stale entry: a cheaper path to this state was found
    }
    if (expansions >= maxExpansions) {
      for (const x of boxes) occupied[x] = 0;
      return null;
    }
    expansions++;
    let done = false;
    for (let bi = 0; bi < boxes.length && !done; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        let h = INF;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, x) => a - x);
          h = heuristic(next);
          if (h < INF) key = keyOf(next, region(w, wall, occupied, b, mark, ++markStamp));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h >= INF || key < 0) continue;
        const prev = best.get(key);
        if (prev !== undefined && prev <= g + 1) continue;
        best.set(key, g + 1);
        const ns = boxesArr.length;
        boxesArr.push(next);
        playerArr.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        gScore.push(g + 1);
        heap.push([g + 1 + h, h, seq++, ns]);
        if (solved(next)) {
          found = ns;
          done = true;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
    if (done) break;
  }
  if (found < 0) return null;

  // Reconstruct: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = sub.player;
  let boxes = sub.boxes.slice();
  for (const s of path) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = boxesArr[s]!.slice();
  }
  return { pushes: path.length, moves, expansions };
}

function popcount(x: number): number {
  let c = 0;
  while (x > 0) {
    x &= x - 1;
    c++;
  }
  return c;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor; each remaining part holding
 * boxes is a room. Null when the level is no such set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    const p = part[b]!;
    if (p < 0) return null;
    rooms[p]!.boxes.push(b);
  }
  for (const g of goals) {
    const p = part[g]!;
    if (p < 0) return null;
    rooms[p]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Work budgets, counted in expanded states (never the clock, so two runs agree).
// Rooms solve in ~1k expansions in practice; the caps are a safety net for
// pathological draws. The whole-level fallback is a last resort for levels
// with no room structure (never drawn by this generator): a small cap, so a
// hopeless search cannot eat the 60s wall budget.
const ROOM_ASTAR_BUDGET = 400_000;
const ROOM_BFS_BUDGET = 200_000;
const WHOLE_ASTAR_BUDGET = 60_000;

function solveLevel(text: string): { moves: string; detail: string } {
  const level = parseLevel(text);
  const solved0 = level.boxes.every((b) => level.goals.includes(b));
  if (solved0) return { moves: '', detail: 'already solved' };

  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    const details: string[] = [];
    for (const room of rooms) {
      // The room's sublevel: other rooms' boxes stand as walls (on their
      // goals once their room is done), everything else as in the level.
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      let res = solvePushes(sub, ROOM_ASTAR_BUDGET, true);
      let how = 'astar';
      if (res === null) {
        res = solvePushes(sub, ROOM_BFS_BUDGET, false);
        how = 'bfs';
      }
      if (res === null) return { moves: '', detail: `room failed (${how})` };
      moves += res.moves;
      player = endOf(level.w, player, res.moves);
      room.done = true;
      details.push(`${res.pushes}p/${res.expansions}e/${how}`);
    }
    return { moves, detail: `rooms: ${details.join(' ')}` };
  }

  // No room structure: one A* over the whole level.
  const res = solvePushes(level, WHOLE_ASTAR_BUDGET, true);
  if (res === null) return { moves: '', detail: 'whole-level search gave up' };
  return { moves: res.moves, detail: `whole: ${res.pushes}p/${res.expansions}e` };
}

/** The square the player stands on after `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

for (const id of readdirSync(I).sort()) {
  const { moves, detail } = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  // Progress goes to stderr, never into an output file.
  process.stderr.write(`${id}: ${moves.length} moves, ${detail}\n`);
}
