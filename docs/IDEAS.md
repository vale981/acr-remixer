<!--
SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
SPDX-License-Identifier: MIT
-->

# Ideas and open questions

Things that were looked into but not built (yet), with what is known. Pick
one up, or open an issue to discuss it first.

## Stages from several levels

Today every stage is built from **one** source level: its pieces are cut,
moved, repeated and joined, and a new level is written from that level's
cells, terrain tiles, collision and tables. Mixing pieces of two stages, or
chaining whole stages, needs content from several levels in one world.

The install has two stages per rally (Wales: Hafren North + South; Greece:
Elatia + Loutraki; Alsace: Munster + Saverne; Monte Carlo: Bollene +
Sisteron; Sweden: Vargasen + Fredriksberg).

What it would take:

| Part | Same rally | Different rallies |
|---|---|---|
| Pieces tagged with their source (`HafrenNorth:3-6 HafrenSouth:7-11`) | moderate | same |
| Cells of two levels in one new level: rename both, merge the runtime spatial hash (the cell copies for repeats are a template) | core work | same |
| Main level: sky, lighting, start/finish and route actors exist once; one stage hosts, the other's actors are appended or dropped | hardest part | same |
| Collision, terrain tiles, foliage: existing code, run per source | mostly there | same |
| Environment (sky, light, weather, tyres follow the host rally) | identical | does not fit: foreign terrain under the host's light, gravel next to snow |
| Joins between stages (road width, terrain style) | similar | visible seams |

Estimate: several days for two stages of the same rally, with risk (the
engine is picky about World Partition data, as the cell copies showed).
Suggested first milestone: Hafren North + South back to back (two big pieces,
one join, no new search).

Chaining **all** stages is the same work; only one environment can be active,
so most of the route would look wrong, and the build would exceed 10 GB.
Chaining with a loading screen in between is what the game's own rally mode
already does.

### Runtime alternatives

- **Dynamic unloading** is already what World Partition does while driving;
  memory is not the limit, getting two levels' content into one world is.
- **Teleport** from the end of stage A to the start of stage B (both in one
  world) would avoid the join, but still needs the merging above, needs UE4SS
  on players' PCs (Remixer works with plain mod files today), and the game's
  stage logic (route spline, distance, timer, respawn, finish) expects one
  continuous route.
- **Load stage B at runtime as a level instance** (`LoadLevelInstance`) at an
  offset next to stage A, near A's finish. If this works for World Partition
  levels, no merging would be needed at all. Unreal may not support runtime
  level instances of World Partition maps; untested. Cheap experiment (about an
  hour with UE4SS): while driving Hafren South, load Hafren North as an
  instance and check whether its cells appear.

## Other open items

- **Barrier tape** is hidden for now (`WPConfig.tapes = "hide"`): each tape
  hangs from one pole and ends at the next, so poles moved by different
  pieces (or parked) stretch it across the map. Proper fix: move pole chains
  together and cut them at piece joins.
- **Terrain walls with close repeats:** when the same piece lies several
  times side by side at different heights, the cut edge of the higher terrain
  shows as a steep wall in the distance. Seam blending (60 m) isn't enough;
  ideas: lower the higher terrain over a wider zone, or filler ground in front.
- **Compressed stage containers** (zlib: 8 km stage 0.87 -> 0.42 GB) read back
  correctly, but ACR returns to the main menu when loading them, even with the
  container and per-chunk Compressed flags and raw incompressible blocks. No
  game log (shipping build). Guess: ACR accepts only Oodle in IoStore, which
  can't be written here. Retry only with a new idea.
- **Error when leaving a stage** reported by one user (UE error on exit); not
  reproducible here, details requested.
