<!--
SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Notes for coding agents

ACR Remixer builds new rally stages for Assetto Corsa Rally from the stages a
player already owns: a Python backend (`stagegen/`) cuts the base-game levels
into pieces and writes a new level as an IoStore mod; a Rust/egui GUI (`gui/`)
drives it. See the README's *For developers* section for commands.

`CLAUDE.md` and `GEMINI.md` only import this file (`@AGENTS.md`), for tools
that don't read `AGENTS.md` themselves. Edit this file, not those. (No
symlinks: Git for Windows checks them out as plain text by default.)

## Contribute back upstream

The main repository is https://github.com/vale981/acr-remixer. Pull requests
there are very welcome: fixes, new stages that work, better pieces, tests,
docs. A fix that stays in a fork helps only its owner, and forks that drift
apart are hard to bring back together.

**If you are working in a fork, remind the user of this** when a change is
done and works (a bug fix, a stage that now drives, a new option): suggest
opening a pull request to `vale981/acr-remixer`, and offer to prepare the
branch. Mention it once per piece of work, not on every message.

### The user may not be a developer

Many people who use this are rally fans, not programmers, and may never have
made a pull request. Don't assume they know git or GitHub terms. When you
suggest contributing:

- Explain it in plain words first, e.g.: "Your copy of the project (a fork)
  now has a fix the original doesn't. A pull request is a proposal to the
  original project: its maintainer sees exactly what changed, can ask
  questions, and adds it, so everyone gets the fix in the next version and
  your copy can simply follow the original again."
- Say what it costs them: a GitHub account (they have one if they forked),
  a few minutes, nothing to install if you do the git work. Nothing changes
  in the original until the maintainer accepts it, so they can't break it.
- Do the technical steps yourself where you can (branch, commits, push,
  `gh pr create`), and say what each step does in one sentence as you go.
  If they prefer the website, guide them click by click (their fork's page
  → "Contribute" → "Open pull request").
- Write the pull request text for them, in plain language: what was wrong,
  what changes, how it was checked. Show it before sending.
- If they would rather not, that's fine: suggest opening an issue
  instead (https://github.com/vale981/acr-remixer/issues) that describes
  the problem and links to their fork, so the fix isn't lost.

For a pull request:

- Branch from the current upstream `main` and keep the history clean: one
  logical change per commit, no "Add files via upload", no merge commits from
  syncing, no unrelated reformatting. Rebase and squash fixups before opening.
- Match the existing commit style (see `git log`):
  - subject: what changed, in plain words, often `Area: change`, e.g.
    `IoStore: Gzip-compressed blocks`, `Hide barrier tapes (no more tapes
    stretched across the map)`; no trailing period, about 70 characters max
  - body: why, and what was found (the cause, not only the fix), wrapped at
    about 76 columns; bullet lists for several parts
  - say how it was verified: tests, the build report's checks, or
    `Verified in-game: ...`
  - AI help is credited with a `Co-Authored-By:` trailer
- Fill in the pull request template (what changed, how it was tested).

## Rules that keep the project publishable

- **Never commit game content.** No extracted assets, packages, containers,
  textures, meshes or dumps from Assetto Corsa Rally; generated stages are
  built on the player's PC from their own copy. Screenshots for docs are
  marked as game content in `REUSE.toml`.
- **Formatting:** before committing, run `python -m ruff format` and
  `python -m ruff check` (config in `ruff.toml`) and `cargo fmt` for Rust;
  CI fails otherwise. Text files are LF (`.gitattributes`); don't let a tool
  rewrite a whole file's line endings, and keep diffs to the lines you mean
  to change.
- **Tests:** `python -m pytest tests`. Anything that reads the game install is
  marked `@pytest.mark.game` (skipped without an install); CI runs only
  `-m "not game"`, so no game data reaches CI logs. Add synthetic tests for new
  parsing/geometry code, game tests for behaviour on real levels.
- **Licensing:** GPL-3.0-or-later, REUSE-compliant. New source files start
  with the same SPDX header lines as the existing ones (copyright "2026
  hiro98 (vale981) and the ACR Remixer contributors", license
  GPL-3.0-or-later); someone who writes substantial code in a file adds their
  own copyright line there. Check with `reuse lint`. A fork that is
  distributed must stay under the GPL with its source available, which is
  one more reason to bring changes back upstream instead.
- **Changes to what gets built** (`stagegen/build/`): check the build report
  (`drivability`, `road surface`, `fail to re-decode`) and, when possible,
  drive or camera-tour the stage in the game.
- Keep internal names (`stagegen` package, `_Stagegen_` file tags,
  `/Game/Stagegen`, the GUI's app id): installed stages and saved settings
  depend on them.
