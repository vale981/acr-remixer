// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-FileCopyrightText: 2026 iceone88
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

//! ACR Remixer GUI (egui; formerly acr-stagegen). Phase 1: drives the Python pipeline
//! (`python -m stagegen.cli build ...`) and previews the result; the pipeline is
//! being ported to Rust module by module.
#![windows_subsystem = "windows"]

mod steam;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use serde::{Deserialize, Serialize};

const STAGES: &[(&str, &str)] = &[
    ("WalesS4HafrenSouth", "Wales – Hafren South"),
    ("WalesS3HafrenNorth", "Wales – Hafren North"),
    ("SwedenS1Vargasen", "Sweden – Vargåsen"),
    ("SwedenS3Fredriksberg", "Sweden – Fredriksberg"),
    ("MonteCarloS1Bollene", "Monte Carlo – Bollène"),
    ("MonteCarloS2Sisteron", "Monte Carlo – Sisteron"),
    ("GreeceS3Elatia", "Greece – Elatia"),
    ("GreeceS4Loutraki", "Greece – Loutraki"),
    ("AlsaceS2Munster", "Alsace – Munster"),
    ("AlsaceS4Saverne", "Alsace – Saverne"),
];
/// Stages verified in-game with the hands-off test; the others are experimental.
const VERIFIED: &[&str] = &["WalesS4HafrenSouth", "GreeceS3Elatia"];

#[cfg(embedded_backend)]
const BACKEND: Option<&[u8]> = Some(include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../packaging/dist/stagegen-backend.exe"
)));
#[cfg(not(embedded_backend))]
const BACKEND: Option<&[u8]> = None;

/// Unpack the embedded backend once (the name carries a content hash, so a
/// new release replaces it) and return its path.
fn embedded_backend() -> Result<PathBuf, String> {
    let bytes = BACKEND.ok_or("this build has no embedded backend")?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes.iter().step_by(64) {
        h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
    }
    let dir = PathBuf::from(std::env::var("LOCALAPPDATA").map_err(|e| e.to_string())?)
        .join("acr-stagegen");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let exe = dir.join(format!("stagegen-backend-{:x}-{:016x}.exe", bytes.len(), h));
    if !exe.is_file() {
        let tmp = exe.with_extension("tmp");
        std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &exe).map_err(|e| e.to_string())?;
    }
    Ok(exe)
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Prefs {
    acr_dir: String,
    python: String,
    backend_dir: String,
    out_dir: String,
    stage: String,
    length_m: f32,
    seed: u32,
    batch: u32,
    /// how often a piece of road may appear (>1: longer stages from repeats)
    max_uses: u32,
    /// build `sequence` (pieces like "0-1 7-11 2-7r 18-19") instead of searching
    use_sequence: bool,
    sequence: String,
    /// a share code (R2:...) to rebuild instead of searching; empty = use the form
    code: String,
    forest: bool,
    grass: bool,
    foliage: bool,
    install: bool,
    launch: bool,
    /// development: run `python -m stagegen.cli` from backend_dir instead of the embedded backend
    use_python: bool,
    /// block ACR's network access (Windows Firewall) while a generated stage is installed
    block_online: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        let acr = steam::detect_acr()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let py = std::env::var("LOCALAPPDATA")
            .map(|l| PathBuf::from(l).join(r"Programs\Python\Python314\python.exe"))
            .ok()
            .filter(|p| p.is_file())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "python".into());
        let backend = if acr.is_empty() {
            String::new()
        } else {
            format!(r"{acr}\acr-stagegen")
        };
        let out = if backend.is_empty() {
            String::new()
        } else {
            format!(r"{backend}\out")
        };
        Prefs {
            acr_dir: acr,
            python: py,
            backend_dir: backend,
            out_dir: out,
            stage: STAGES[0].0.into(),
            length_m: 4500.0,
            seed: 1,
            batch: 1,
            max_uses: 1,
            use_sequence: false,
            sequence: String::new(),
            code: String::new(),
            forest: true,
            grass: true,
            foliage: true,
            install: true,
            launch: false,
            use_python: BACKEND.is_none(),
            block_online: false,
        }
    }
}

/// A generated stage installed in the game's ~mods folder.
#[derive(Clone)]
struct Installed {
    /// level name (what `uninstall --stage` takes)
    name: String,
    /// menu name; None for an old build that replaces the original stage
    display: Option<String>,
    /// how it was built (level, length, seed, max_uses, sequence, foliage); None for older builds
    params: Option<serde_json::Value>,
    /// its map (layout.json copied into ~mods at install), if there
    layout: Option<PathBuf>,
}

/// Settings with an installed stage's build parameters applied.
fn prefs_with_params(base: &Prefs, params: &serde_json::Value) -> Prefs {
    let mut p = base.clone();
    if let Some(s) = params["level"].as_str() {
        p.stage = s.to_string();
    }
    if let Some(v) = params["length"].as_f64() {
        p.length_m = v as f32;
    }
    if let Some(v) = params["seed"].as_u64() {
        p.seed = v as u32;
    }
    p.max_uses = params["max_uses"].as_u64().unwrap_or(1) as u32;
    p.sequence = params["sequence"].as_str().unwrap_or("").to_string();
    p.use_sequence = !p.sequence.trim().is_empty();
    p.forest = params["forest"].as_bool().unwrap_or(true);
    p.grass = params["grass"].as_bool().unwrap_or(true);
    p.foliage = params["foliage"].as_bool().unwrap_or(true);
    // stages built since share codes exist rebuild from their code (exactly the same pieces)
    p.code = params["code"].as_str().unwrap_or("").to_string();
    p
}

/// Generated stages installed in ~mods (the shared menu-table container excluded).
fn installed_stages(acr_dir: &str) -> Vec<Installed> {
    let mods = PathBuf::from(acr_dir).join(r"acr\Content\Paks\~mods");
    let Ok(rd) = std::fs::read_dir(&mods) else {
        return vec![];
    };
    let mut v: Vec<Installed> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            let name = n
                .strip_prefix("pakchunk99-")?
                .strip_suffix("_Stagegen_9_P.utoc")?
                .to_string();
            if name == "Tables" {
                return None;
            }
            let meta = mods.join(format!("pakchunk99-{name}_Stagegen_9_P.stagegen.json"));
            let j = std::fs::read_to_string(meta)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
            let display = j
                .as_ref()
                .and_then(|j| j["display"].as_str().map(|s| s.to_string()));
            let params = j
                .as_ref()
                .map(|j| j["params"].clone())
                .filter(|p| p.is_object());
            let layout = Some(mods.join(format!("pakchunk99-{name}_Stagegen_9_P.layout.json")))
                .filter(|p| p.is_file());
            Some(Installed {
                name,
                display,
                params,
                layout,
            })
        })
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// Sequence text as the backend normalizes it (single spaces, no commas).
fn norm_sequence(s: &str) -> String {
    s.replace(',', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tag of a piece sequence: FNV-1a 32-bit of the normalized text (= backend `_seq_tag`).
fn fnv_tag(text: &str) -> String {
    let mut h: u32 = 0x811C_9DC5;
    for b in text.bytes() {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    format!("{h:08X}")
}

fn seq_tag(s: &str) -> String {
    fnv_tag(&norm_sequence(s))
}

/// A share code without whitespace (as the backend tags it).
fn norm_code(code: &str) -> String {
    code.split_whitespace().collect()
}

/// The source level named in an R2 share code.
fn code_level(code: &str) -> Option<String> {
    let c = norm_code(code);
    let mut parts = c.split(':');
    (parts.next()?.eq_ignore_ascii_case("R2")).then_some(())?;
    parts
        .next()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
}

/// Output folder of a build (matches the backend's naming).
fn build_dir(p: &Prefs, seed: u32) -> PathBuf {
    let base = PathBuf::from(&p.out_dir);
    if !p.code.trim().is_empty() {
        return base.join(format!("{}_code{}", p.stage, fnv_tag(&norm_code(&p.code))));
    }
    if p.use_sequence && !p.sequence.trim().is_empty() {
        return base.join(format!("{}_seq{}", p.stage, seq_tag(&p.sequence)));
    }
    let uses = if p.max_uses > 1 {
        format!("_x{}", p.max_uses)
    } else {
        String::new()
    };
    base.join(format!("{}_seed{seed}_{:.0}m{uses}", p.stage, p.length_m))
}

const FW_RULE: &str = "acr-stagegen: block ACR online";

fn quiet(cmd: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000) // CREATE_NO_WINDOW
}

/// Is the firewall rule that blocks ACR's network access present?
fn fw_blocked() -> bool {
    quiet(Command::new("netsh").args([
        "advfirewall",
        "firewall",
        "show",
        "rule",
        &format!("name={FW_RULE}"),
    ]))
    .output()
    .map(|o| o.status.success())
    .unwrap_or(false)
}

/// Add (in + out) or remove the blocking rule for acr.exe; one UAC prompt.
fn fw_set(acr_dir: &str, block: bool) -> Result<(), String> {
    let exe = PathBuf::from(acr_dir).join(r"acr\Binaries\Win64\acr.exe");
    let mut cmds = format!("netsh advfirewall firewall delete rule name=\"{FW_RULE}\"");
    if block {
        if !exe.is_file() {
            return Err(format!("{} not found", exe.display()));
        }
        for dir in ["out", "in"] {
            cmds.push_str(&format!(
                " & netsh advfirewall firewall add rule name=\"{FW_RULE}\" dir={dir} action=block program=\"{}\" enable=yes",
                exe.display()
            ));
        }
    }
    let ps = format!(
        "Start-Process cmd -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '/c {}'",
        cmds.replace('\'', "''")
    );
    let st = quiet(Command::new("powershell").args(["-NoProfile", "-Command", &ps]))
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err("cancelled or no admin rights".into());
    }
    if fw_blocked() != block {
        return Err("the firewall rule did not change (admin prompt declined?)".into());
    }
    Ok(())
}

const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("GIT_HASH"), ")");

/// Hide the Windows user name in paths before text leaves the PC.
fn redact(s: &str) -> String {
    let mut out = s.to_string();
    if let Ok(home) = std::env::var("USERPROFILE") {
        out = out.replace(&home, "%USERPROFILE%");
    }
    if let Ok(user) = std::env::var("USERNAME") {
        if user.len() >= 3 {
            out = out.replace(&user, "<user>");
        }
    }
    out
}

/// Short code for stage + length + seed + foliage: "WalesS4HafrenSouth/4500/939252/FGL",
/// plus "/x2" for repeats and "/0-1_7-11_7-11r_18-19" for a sequence when used.
fn share_code(p: &Prefs) -> String {
    let fol: String = [(p.forest, 'F'), (p.grass, 'G'), (p.foliage, 'L')]
        .iter()
        .map(|(on, c)| if *on { *c } else { '-' })
        .collect();
    let mut code = format!("{}/{:.0}/{}/{}", p.stage, p.length_m, p.seed, fol);
    if p.use_sequence && !p.sequence.trim().is_empty() {
        code += &format!("/{}", norm_sequence(&p.sequence).replace(' ', "_"));
    } else if p.max_uses > 1 {
        code += &format!("/x{}", p.max_uses);
    }
    code
}

fn apply_share_code(p: &mut Prefs, code: &str) -> Result<(), String> {
    let parts: Vec<&str> = code.trim().split('/').collect();
    if parts.len() != 4 && parts.len() != 5 {
        return Err(
            "expected Stage/Length/Seed/Foliage, e.g. WalesS4HafrenSouth/4500/1/FGL".into(),
        );
    }
    p.max_uses = 1;
    p.use_sequence = false;
    if let Some(extra) = parts.get(4) {
        if let Some(n) = extra.strip_prefix('x') {
            p.max_uses = n.parse::<u32>().map_err(|_| "bad repeats")?.clamp(1, 5);
        } else {
            p.sequence = extra.replace('_', " ");
            p.use_sequence = true;
        }
    }
    if !STAGES.iter().any(|s| s.0 == parts[0]) {
        return Err(format!("unknown stage {:?}", parts[0]));
    }
    let len: f32 = parts[1].parse().map_err(|_| "bad length")?;
    let seed: u32 = parts[2].parse().map_err(|_| "bad seed")?;
    p.stage = parts[0].to_string();
    p.length_m = len.max(1000.0);
    p.seed = seed.max(1);
    p.forest = parts[3].contains('F');
    p.grass = parts[3].contains('G');
    p.foliage = parts[3].contains('L');
    Ok(())
}

/// Summary of the newest ACR crash reports (error + top of the call stack).
fn acr_crashes(n: usize) -> String {
    let Ok(base) = std::env::var("LOCALAPPDATA") else {
        return "(no LOCALAPPDATA)\n".into();
    };
    let dir = PathBuf::from(base).join(r"acr\Saved\Crashes");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return "(no crash reports)\n".into();
    };
    let mut ds: Vec<_> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    ds.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = String::new();
    for (t, d) in ds.into_iter().take(n) {
        let age = t.elapsed().map(|e| e.as_secs() / 60).unwrap_or(0);
        let txt = std::fs::read_to_string(d.join("CrashContext.runtime-xml")).unwrap_or_default();
        let tag = |name: &str| -> String {
            let (a, b) = (format!("<{name}>"), format!("</{name}>"));
            txt.find(&a)
                .and_then(|i| {
                    txt[i + a.len()..]
                        .find(&b)
                        .map(|j| txt[i + a.len()..i + a.len() + j].trim().to_string())
                })
                .unwrap_or_default()
        };
        let stack: Vec<String> = tag("PCallStack")
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .take(15)
            .collect();
        out.push_str(&format!(
            "--- crash {} min ago: {}\n{}\n",
            age,
            tag("ErrorMessage"),
            stack.join("\n")
        ));
    }
    if out.is_empty() {
        "(no crash reports)\n".into()
    } else {
        out
    }
}

fn windows_version() -> String {
    quiet(Command::new("cmd").args(["/c", "ver"]))
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "?".into())
}

/// The finished route of a built stage (from layout.json).
struct Preview {
    name: String,
    pts: Vec<[f64; 3]>,
    s: Vec<f64>,
    joins: Vec<f64>,
    length: f64,
    notes: usize,
    /// its piece sequence ("0-1 7-11 2-7r 18-19"), to edit and rebuild
    sequence: String,
    /// its share code (R2:...), empty for stages built before share codes
    share_code: String,
}

/// A tile-grid crossing on the untouched source-stage route.
#[derive(Clone, Deserialize)]
struct SourceCut {
    i: usize,
    kind: String,
    x: f64,
    y: f64,
}

/// One valid source interval that can be referenced by a manual sequence.
#[derive(Clone, Deserialize)]
struct SourcePiece {
    id: String,
    a: usize,
    b: usize,
    length: f64,
    reversible: bool,
    /// True for an elementary map section when this exact interval is also
    /// accepted by the backend's manual-sequence candidate list.
    buildable: bool,
    station_start: f64,
    station_end: f64,
    station: Vec<f64>,
    midpoint: [f64; 3],
    elevation_start: f64,
    elevation_end: f64,
    elevation_min: f64,
    elevation_max: f64,
    xyz: Vec<[f64; 3]>,
}

/// Read-only preview data returned by `stagegen segments --json`.
#[derive(Clone, Deserialize)]
struct SourcePreview {
    level: String,
    route_xyz: Vec<[f64; 3]>,
    route_s: Vec<f64>,
    cuts: Vec<SourceCut>,
    /// Every consecutive cut-to-cut section, whether or not it is buildable.
    sections: Vec<SourcePiece>,
    /// All candidate intervals accepted by make_ranges(), including multi-cut spans.
    pieces: Vec<SourcePiece>,
}

/// `load_wp_source()` may print progress/diagnostic lines before the JSON.
/// The backend now emits a clean document for --json; selecting the last
/// top-level JSON line also keeps the GUI tolerant of older development builds.
fn parse_source_preview(stdout: &str) -> Result<SourcePreview, String> {
    let document = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with('{'))
        .ok_or_else(|| "backend did not emit a JSON stock-stage description".to_string())?;
    serde_json::from_str::<SourcePreview>(document).map_err(|e| e.to_string())
}

fn parse_piece_token(token: &str) -> Option<(usize, usize, bool)> {
    let reversed = token.ends_with('r');
    let core = if reversed {
        &token[..token.len().saturating_sub(1)]
    } else {
        token
    };
    let (a, b) = core.split_once('-')?;
    let a = a.parse::<usize>().ok()?;
    let b = b.parse::<usize>().ok()?;
    if a >= b {
        return None;
    }
    Some((a, b, reversed))
}

/// Append a source piece only when its entry cut matches the current sequence exit.
/// This checks topological continuity; the backend still performs full geometric
/// kink, grade, width and clearance validation when the sequence is built.
fn append_piece_to_sequence(
    sequence: &str,
    piece: &SourcePiece,
    final_cut: usize,
) -> Result<String, String> {
    let current = norm_sequence(sequence);
    if current.is_empty() {
        if piece.a != 0 {
            return Err(format!(
                "Start the manual sequence with a piece beginning at cut 0, not {}.",
                piece.id
            ));
        }
        return Ok(piece.id.clone());
    }

    let last_token = current
        .split_whitespace()
        .last()
        .ok_or("The sequence is empty.")?;
    let (a, b, reversed) = parse_piece_token(last_token)
        .ok_or_else(|| format!("Cannot understand the last sequence piece: {last_token}"))?;
    let exit = if reversed { a } else { b };
    if exit == final_cut {
        return Err("The sequence already reaches the finish cut; remove its last piece before adding another.".into());
    }

    let token = if piece.a == exit {
        piece.id.clone()
    } else if piece.b == exit && piece.reversible {
        format!("{}r", piece.id)
    } else {
        return Err(format!(
            "Piece {} does not connect to the current sequence exit (cut {}). Choose a piece starting or ending at cut {}.",
            piece.id, exit, exit));
    };

    // Validate that the existing text contains a continuous chain before appending.
    let mut expected_entry = 0usize;
    for item in current.split_whitespace() {
        let (ia, ib, rev) = parse_piece_token(item)
            .ok_or_else(|| format!("Cannot understand sequence piece: {item}"))?;
        let (entry, next) = if rev { (ib, ia) } else { (ia, ib) };
        if entry != expected_entry {
            return Err(format!(
                "Existing sequence is discontinuous before {item} (expected cut {expected_entry})."
            ));
        }
        expected_entry = next;
    }
    if expected_entry != exit {
        return Err("The last sequence piece does not end at the expected cut.".into());
    }
    Ok(format!("{current} {token}"))
}

fn point_segment_distance(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let denom = ab.length_sq();
    if denom <= f32::EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(ab) / denom).clamp(0.0, 1.0);
    point.distance(a + t * ab)
}

/// Pick the closest physical cut-to-cut section under the pointer.
/// All map sections are inspectable, including those excluded from generation;
/// only entries also present in `source.pieces` can be added to a sequence.
fn source_piece_at_pointer(pu: &egui_plot::PlotUi<'_>, source: &SourcePreview) -> Option<String> {
    let pointer = pu.ctx().input(|i| i.pointer.interact_pos())?;
    let mut best: Option<(f32, f32, String)> = None;
    for piece in &source.sections {
        if piece.xyz.len() < 2 {
            continue;
        }
        let screen: Vec<egui::Pos2> = piece
            .xyz
            .iter()
            .map(|p| pu.screen_from_plot(egui_plot::PlotPoint::new(p[0], p[1])))
            .collect();
        let distance = screen
            .windows(2)
            .map(|w| point_segment_distance(pointer, w[0], w[1]))
            .fold(f32::INFINITY, f32::min);
        if distance > 18.0 {
            continue;
        }
        let midpoint = pu.screen_from_plot(egui_plot::PlotPoint::new(
            piece.midpoint[0],
            piece.midpoint[1],
        ));
        let midpoint_distance = midpoint.distance(pointer);
        let replace = match &best {
            None => true,
            Some((old_distance, old_midpoint, _)) => {
                distance < *old_distance - 0.5
                    || ((distance - *old_distance).abs() <= 0.5
                        && midpoint_distance < *old_midpoint)
            }
        };
        if replace {
            best = Some((distance, midpoint_distance, piece.id.clone()));
        }
    }
    best.map(|(_, _, id)| id)
}

/// The generated route split at the joins: (point, station) per piece, each
/// overlapping its neighbours by 10 m so the coloured lines meet.
fn piece_spans(pv: &Preview) -> Vec<Vec<([f64; 3], f64)>> {
    let mut bounds = vec![0.0];
    bounds.extend(pv.joins.iter().copied());
    bounds.push(f64::INFINITY);
    bounds
        .windows(2)
        .map(|w| {
            pv.pts
                .iter()
                .zip(&pv.s)
                .filter(|(_, s)| **s >= w[0] - 10.0 && **s <= w[1] + 10.0)
                .map(|(p, s)| (*p, *s))
                .collect()
        })
        .collect()
}

/// Legend name of the k-th piece of a generated stage: its place and source
/// piece ("2. 7-11r"), numbered because a piece can appear more than once.
fn piece_label(pv: &Preview, k: usize) -> String {
    match pv.sequence.split_whitespace().nth(k) {
        Some(id) => format!("{}. {id}", k + 1),
        None => format!("piece {}", k + 1),
    }
}

/// Height profile of a generated stage, each piece in its map colour (the
/// colour changes at the joins).
fn preview_elevation_ui(ui: &mut egui::Ui, pv: &Preview, height: f32) {
    Plot::new("elev")
        .height(height)
        .y_axis_label("m")
        .show(ui, |pu| {
            for (k, pts) in piece_spans(pv).into_iter().enumerate() {
                let pts: Vec<[f64; 2]> = pts.iter().map(|(p, s)| [*s, p[2]]).collect();
                pu.line(
                    Line::new(PlotPoints::from(pts))
                        .width(2.5_f32)
                        .color(section_color(k, true)),
                );
            }
        });
}

/// Highlight of the selected section (map and elevation).
const SELECTED: egui::Color32 = egui::Color32::from_rgb(245, 190, 55);

/// Colour of the k-th cut-to-cut section, the same on the map and the elevation.
fn section_color(k: usize, buildable: bool) -> egui::Color32 {
    const PALETTE: [egui::Color32; 8] = [
        egui::Color32::from_rgb(68, 136, 205),
        egui::Color32::from_rgb(66, 166, 130),
        egui::Color32::from_rgb(205, 151, 72),
        egui::Color32::from_rgb(155, 117, 205),
        egui::Color32::from_rgb(205, 103, 119),
        egui::Color32::from_rgb(68, 165, 181),
        egui::Color32::from_rgb(178, 180, 82),
        egui::Color32::from_rgb(194, 119, 73),
    ];
    if buildable {
        PALETTE[k % PALETTE.len()]
    } else {
        egui::Color32::from_gray(95)
    }
}

fn source_map_ui(
    ui: &mut egui::Ui,
    source: &SourcePreview,
    selected: Option<&str>,
    height: f32,
) -> Option<String> {
    ui.label(format!(
        "Stock stage: {}  •  {} tile sections  •  {} buildable candidates",
        source.level,
        source.sections.len(),
        source.pieces.len()
    ));
    ui.small("Click a section on the map or the height profile to inspect it. Coloured sections are valid standalone candidates; gray sections are shown for reference but are excluded by build filters. Longer candidates are listed below.");

    let mut clicked_piece = None;
    Plot::new("stock-map")
        .data_aspect(1.0)
        .legend(Legend::default())
        .height(height)
        .show(ui, |pu| {
            let route: Vec<[f64; 2]> = source.route_xyz.iter().map(|p| [p[0], p[1]]).collect();
            if route.len() >= 2 {
                pu.line(
                    Line::new(PlotPoints::from(route))
                        .width(1.5_f32)
                        .color(egui::Color32::from_gray(120))
                        .name("stock centreline"),
                );
            }

            // Draw every physical cut-to-cut interval, including those excluded
            // from generation by the minimum-length or source-geometry filters.
            for (k, section) in source.sections.iter().enumerate() {
                let pts: Vec<[f64; 2]> = section.xyz.iter().map(|p| [p[0], p[1]]).collect();
                if pts.len() >= 2 {
                    // no legend entry: a stage has up to ~40 sections (the
                    // legend would cover the map); each is labelled on the map
                    let color = section_color(k, section.buildable);
                    pu.line(Line::new(PlotPoints::from(pts)).width(3.0_f32).color(color));
                    pu.text(
                        egui_plot::Text::new(
                            egui_plot::PlotPoint::new(section.midpoint[0], section.midpoint[1]),
                            section.id.clone(),
                        )
                        .color(if section.buildable {
                            color
                        } else {
                            egui::Color32::from_gray(175)
                        }),
                    );
                }
            }

            if let Some(id) = selected {
                if let Some(piece) = find_source_piece(source, id) {
                    let pts: Vec<[f64; 2]> = piece.xyz.iter().map(|p| [p[0], p[1]]).collect();
                    if pts.len() >= 2 {
                        pu.line(
                            Line::new(PlotPoints::from(pts))
                                .width(5.0_f32)
                                .color(SELECTED)
                                .name(format!("selected {id}")),
                        );
                        pu.points(
                            egui_plot::Points::new(vec![[piece.midpoint[0], piece.midpoint[1]]])
                                .radius(5.0_f32)
                                .color(SELECTED)
                                .name("selected midpoint"),
                        );
                    }
                }
            }

            let cut_points: Vec<[f64; 2]> = source.cuts.iter().map(|c| [c.x, c.y]).collect();
            if !cut_points.is_empty() {
                pu.points(
                    egui_plot::Points::new(cut_points)
                        .radius(3.5_f32)
                        .color(egui::Color32::WHITE)
                        .name("tile crossings"),
                );
            }
            for cut in &source.cuts {
                let label = match cut.kind.as_str() {
                    "start" => format!("{} START", cut.i),
                    "finish" => format!("{} FINISH", cut.i),
                    _ => cut.i.to_string(),
                };
                pu.text(
                    egui_plot::Text::new(egui_plot::PlotPoint::new(cut.x, cut.y), label).color(
                        if cut.kind == "start" || cut.kind == "finish" {
                            egui::Color32::from_rgb(245, 190, 55)
                        } else {
                            egui::Color32::WHITE
                        },
                    ),
                );
            }
            if pu.response().clicked() {
                clicked_piece = source_piece_at_pointer(pu, source);
            }
        });
    clicked_piece
}

fn source_sections_ui(
    ui: &mut egui::Ui,
    source: &SourcePreview,
    selected: Option<&str>,
    filter: &mut String,
) -> Option<String> {
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.strong(format!("Available manual pieces ({})", source.pieces.len()));
        ui.add(
            egui::TextEdit::singleline(filter)
                .desired_width(150.0)
                .hint_text("Filter e.g. 12-13"),
        );
    });
    if source.pieces.is_empty() {
        ui.small("No candidate pieces passed the current build filters. The stock route and all tile sections are still shown above.");
    }
    egui::ScrollArea::vertical()
        .max_height(190.0)
        .show(ui, |ui| {
            for piece in &source.pieces {
                if !filter.trim().is_empty() && !piece.id.contains(filter.trim()) {
                    continue;
                }
                let direction = if piece.reversible {
                    "reversible"
                } else {
                    "forward only"
                };
                let label = format!(
                    "{:>5}   {:>5.0} m   Z {:.0}–{:.0} m   {}",
                    piece.id, piece.length, piece.elevation_min, piece.elevation_max, direction
                );
                if ui
                    .selectable_label(selected == Some(piece.id.as_str()), label)
                    .clicked()
                {
                    picked = Some(piece.id.clone());
                }
            }
        });
    picked
}

/// The selected piece (a candidate) or map section with this id.
fn find_source_piece<'a>(source: &'a SourcePreview, id: &str) -> Option<&'a SourcePiece> {
    source
        .pieces
        .iter()
        .find(|p| p.id == id)
        .or_else(|| source.sections.iter().find(|p| p.id == id))
}

/// Height profile of the stock stage, each section in its map colour; a click
/// selects the section under the pointer.
fn source_elevation_ui(
    ui: &mut egui::Ui,
    source: &SourcePreview,
    selected: Option<&str>,
    height: f32,
) -> Option<String> {
    let mut clicked = None;
    Plot::new("stock-elevation")
        .height(height)
        .y_axis_label("m")
        .show(ui, |pu| {
            for (k, section) in source.sections.iter().enumerate() {
                let pts: Vec<[f64; 2]> = section
                    .xyz
                    .iter()
                    .zip(&section.station)
                    .map(|(p, s)| [*s, p[2]])
                    .collect();
                if pts.len() >= 2 {
                    pu.line(
                        Line::new(PlotPoints::from(pts))
                            .width(2.5_f32)
                            .color(section_color(k, section.buildable)),
                    );
                }
            }
            if let Some(piece) = selected.and_then(|id| find_source_piece(source, id)) {
                let pts: Vec<[f64; 2]> = piece
                    .xyz
                    .iter()
                    .zip(&piece.station)
                    .map(|(p, s)| [*s, p[2]])
                    .collect();
                if pts.len() >= 2 {
                    pu.line(
                        Line::new(PlotPoints::from(pts))
                            .width(5.0_f32)
                            .color(SELECTED),
                    );
                }
            }
            if pu.response().clicked() {
                if let Some(at) = pu.pointer_coordinate() {
                    clicked = source
                        .sections
                        .iter()
                        .find(|p| p.station_start <= at.x && at.x <= p.station_end)
                        .map(|p| p.id.clone());
                }
            }
        });
    clicked
}

/// The old full-size stock elevation (Elevation tab).
fn source_elevation_tab_ui(
    ui: &mut egui::Ui,
    source: &SourcePreview,
    selected: Option<&str>,
) -> Option<String> {
    ui.label(format!("Stock-stage elevation: {}", source.level));
    if let (Some(first), Some(last)) = (source.route_xyz.first(), source.route_xyz.last()) {
        let (z_min, z_max) = source
            .route_xyz
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), point| {
                (lo.min(point[2]), hi.max(point[2]))
            });
        ui.small(format!(
            "Route length: {:.0} m  •  start/end: {:.1}/{:.1} m  •  elevation range: {:.1}–{:.1} m",
            source.route_s.last().copied().unwrap_or(0.0),
            first[2],
            last[2],
            z_min,
            z_max
        ));
    }
    source_elevation_ui(
        ui,
        source,
        selected,
        (ui.available_height() - 260.0).max(200.0),
    )
}

fn load_preview(dir: &Path) -> Result<Preview, String> {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    load_preview_file(&dir.join("layout.json"), name)
}

fn load_preview_file(file: &Path, name: String) -> Result<Preview, String> {
    let txt = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
    let pts: Vec<[f64; 3]> = v["preview"]
        .as_array()
        .ok_or("layout.json has no preview (rebuild with a current backend)")?
        .iter()
        .filter_map(|p| {
            let a = p.as_array()?;
            Some([
                a.first()?.as_f64()?,
                a.get(1)?.as_f64()?,
                a.get(2)?.as_f64()?,
            ])
        })
        .collect();
    let mut s = vec![0.0];
    for w in pts.windows(2) {
        let d = ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt();
        s.push(s.last().unwrap() + d);
    }
    let joins = v["pieces"]
        .as_array()
        .map(|a| {
            a.iter()
                .skip(1)
                .filter_map(|p| p["s_start"].as_f64())
                .collect()
        })
        .unwrap_or_default();
    let sequence = v["pieces"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let id = p["segment"].as_str()?.rsplit('#').next()?.to_string();
                    Some(if p["reversed"].as_bool().unwrap_or(false) {
                        id + "r"
                    } else {
                        id
                    })
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    Ok(Preview {
        name,
        length: v["length_m"].as_f64().unwrap_or(*s.last().unwrap()),
        notes: v["pacenotes"].as_array().map(|a| a.len()).unwrap_or(0),
        pts,
        s,
        joins,
        sequence,
        share_code: v["share_code"].as_str().unwrap_or("").to_string(),
    })
}

enum Msg {
    Line {
        line: String,
        stdout: bool,
    },
    Done {
        ok: bool,
        out: Option<PathBuf>,
        stdout: String,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum JobKind {
    Build,
    ListPieces,
    Uninstall,
}

struct Job {
    rx: Receiver<Msg>,
    remaining: Vec<u32>,
    total: usize,
    progress: f32,
    phase: String,
    kind: JobKind,
}

struct App {
    prefs: Prefs,
    log: String,
    job: Option<Job>,
    preview: Option<Preview>,
    /// Original stock-stage route and tile-cut pieces, loaded by "List pieces".
    source_preview: Option<SourcePreview>,
    selected_source_piece: Option<String>,
    show_stock: bool,
    section_filter: String,
    sequence_note: Option<String>,
    tab: Tab,
    /// generated stage in ~mods, firewall block present (refreshed on events)
    installed: Vec<Installed>,
    blocked: bool,
    /// share-code text field (paste a code here to load it)
    code_input: String,
    /// settings of the running build job when they are not the form's (regenerate)
    job_prefs: Option<Prefs>,
    /// installed stage shown on the map (level name)
    shown_installed: Option<String>,
    /// feedback next to the share section ("Copied: ...")
    share_note: Option<String>,
}

/// What was clicked in the list of installed stages.
enum InstAction {
    Show(Installed),
    Load(Installed),
    Share(Installed),
    Regenerate(Installed),
    Uninstall(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Map,
    Elevation,
    Log,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let prefs: Prefs = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();
        let mut app = App {
            prefs,
            log: String::new(),
            job: None,
            preview: None,
            source_preview: None,
            selected_source_piece: None,
            show_stock: false,
            section_filter: String::new(),
            sequence_note: None,
            tab: Tab::Map,
            installed: vec![],
            blocked: false,
            code_input: String::new(),
            job_prefs: None,
            shown_installed: None,
            share_note: None,
        };
        app.preview = load_preview(&app.out_for(app.prefs.seed)).ok(); // reopen the last stage
        app.refresh_install_state();
        app
    }

    fn out_for(&self, seed: u32) -> PathBuf {
        build_dir(&self.prefs, seed)
    }

    fn using_sequence(&self) -> bool {
        self.prefs.use_sequence && !self.prefs.sequence.trim().is_empty()
    }

    fn start(&mut self) {
        // a sequence or a share code is one stage
        let n = if self.using_sequence() || !self.prefs.code.trim().is_empty() {
            1
        } else {
            self.prefs.batch.max(1)
        };
        let seeds: Vec<u32> = (0..n).map(|i| self.prefs.seed + i).collect();
        self.log.clear();
        self.tab = Tab::Log;
        let (_, rx) = channel();
        let total = seeds.len();
        self.job_prefs = None;
        self.job = Some(Job {
            rx,
            remaining: seeds,
            total,
            progress: 0.0,
            phase: "starting".into(),
            kind: JobKind::Build,
        });
        self.next();
    }

    /// Build an installed stage again with its own parameters and reinstall it
    /// (e.g. after an update with fixes); the form is left as it is.
    fn regenerate(&mut self, inst: &Installed) {
        let Some(params) = &inst.params else { return };
        let mut p = prefs_with_params(&self.prefs, params);
        p.install = true;
        p.launch = false;
        p.batch = 1;
        self.log.clear();
        self.log.push_str(&format!(
            "regenerating {}\n",
            inst.display.as_deref().unwrap_or(&inst.name)
        ));
        self.tab = Tab::Log;
        let (_, rx) = channel();
        let seed = p.seed;
        self.job_prefs = Some(p);
        self.job = Some(Job {
            rx,
            remaining: vec![seed],
            total: 1,
            progress: 0.0,
            phase: "starting".into(),
            kind: JobKind::Build,
        });
        self.next();
    }

    /// Show an installed stage's map.
    fn show_installed(&mut self, inst: &Installed) {
        let Some(file) = &inst.layout else { return };
        match load_preview_file(
            file,
            inst.display.clone().unwrap_or_else(|| inst.name.clone()),
        ) {
            Ok(pv) => {
                self.preview = Some(pv);
                self.show_stock = false;
                self.shown_installed = Some(inst.name.clone());
                if self.tab == Tab::Log {
                    self.tab = Tab::Map;
                }
            }
            Err(e) => self.log.push_str(&format!("map of {}: {e}\n", inst.name)),
        }
    }

    /// Launch the build for the next seed of the batch (install/launch only on the last).
    fn next(&mut self) {
        let Some(job) = self.job.as_mut() else { return };
        let Some(seed) = (if job.remaining.is_empty() {
            None
        } else {
            Some(job.remaining.remove(0))
        }) else {
            self.job = None;
            return;
        };
        let last = job.remaining.is_empty();
        let p = self.job_prefs.clone().unwrap_or_else(|| self.prefs.clone());
        let mut args: Vec<String> = vec![
            "build".into(),
            "--level".into(),
            p.stage.clone(),
            "--length".into(),
            format!("{:.0}", p.length_m),
            "--seed".into(),
            seed.to_string(),
            "--out".into(),
            p.out_dir.clone(),
        ];
        if !p.code.trim().is_empty() {
            // the code fixes level, pieces and foliage (the backend ignores the rest)
            args.push("--code".into());
            args.push(norm_code(&p.code));
        } else if p.use_sequence && !p.sequence.trim().is_empty() {
            args.push("--sequence".into());
            args.push(norm_sequence(&p.sequence));
        } else if p.max_uses > 1 {
            args.push("--max-uses".into());
            args.push(p.max_uses.to_string());
        }
        if !p.forest {
            args.push("--no-forest".into());
        }
        if !p.grass {
            args.push("--no-grass".into());
        }
        if !p.foliage {
            args.push("--no-foliage".into());
        }
        // every stage of a batch is installed (each gets its own menu entry); ACR starts after the last
        if p.install {
            args.push("--install".into());
        }
        if last && p.install && p.launch {
            args.push("--launch".into());
        }
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks");
        args.push("--paks".into());
        args.push(paks.display().to_string());
        let out = build_dir(&p, seed);
        let (tx, rx) = channel();
        job.rx = rx;
        job.progress = 0.0;
        job.phase = "starting".into();
        self.log
            .push_str(&format!("\n=== {} seed {} ===\n", p.stage, seed));
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(args);
                spawn_build(prog, cwd, full, Some(out), tx);
            }
            Err(e) => {
                self.log.push_str(&format!("backend: {e}\n"));
                self.job = None;
            }
        }
    }

    fn refresh_install_state(&mut self) {
        self.installed = installed_stages(&self.prefs.acr_dir);
        self.blocked = fw_blocked();
    }

    /// Make the firewall block match: on while a generated stage is installed
    /// (if wanted), off otherwise. Asks for admin rights only when it changes.
    fn sync_firewall(&mut self) {
        self.refresh_install_state();
        let want = self.prefs.block_online && !self.installed.is_empty();
        if want != self.blocked {
            match fw_set(&self.prefs.acr_dir, want) {
                Ok(()) => self.log.push_str(if want {
                    "ACR's internet access is blocked while the generated stage is installed.\n"
                } else {
                    "ACR's internet access is allowed again.\n"
                }),
                Err(e) => self.log.push_str(&format!("firewall: {e}\n")),
            }
            self.blocked = fw_blocked();
        }
    }

    /// Write one text file with everything useful for a bug report; returns its path.
    fn export_bug_report(&self) -> Result<PathBuf, String> {
        let p = &self.prefs;
        let mut r = String::new();
        r.push_str(&format!(
            "ACR Remixer bug report\nversion: {VERSION}\nbackend: {}\nwindows: {}\n\n",
            if p.use_python || BACKEND.is_none() {
                "python (development)"
            } else {
                "embedded"
            },
            windows_version()
        ));
        r.push_str(&format!("share code: {}\n", share_code(p)));
        let inst: Vec<&str> = self.installed.iter().map(|i| i.name.as_str()).collect();
        r.push_str(&format!(
            "installed stages: {}\nonline blocked: {}\n\n",
            if inst.is_empty() {
                "none".to_string()
            } else {
                inst.join(", ")
            },
            self.blocked
        ));
        r.push_str("=== settings ===\n");
        r.push_str(&serde_json::to_string_pretty(p).unwrap_or_default());
        r.push_str("\n\n=== last stage ===\n");
        match &self.preview {
            Some(pv) => r.push_str(&format!(
                "{}: {:.2} km, {} pieces (joins at {:?} m), {} pacenotes\n",
                pv.name,
                pv.length / 1000.0,
                pv.joins.len() + 1,
                pv.joins.iter().map(|j| j.round()).collect::<Vec<_>>(),
                pv.notes
            )),
            None => r.push_str("(none loaded)\n"),
        }
        r.push_str("\n=== log of the last run ===\n");
        r.push_str(if self.log.is_empty() {
            "(empty)\n"
        } else {
            &self.log
        });
        r.push_str("\n=== ACR crash reports (newest first) ===\n");
        r.push_str(&acr_crashes(3));
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dir = if Path::new(&p.out_dir).is_dir() {
            PathBuf::from(&p.out_dir)
        } else {
            std::env::temp_dir()
        };
        let path = dir.join(format!("acr-remixer-bugreport-{stamp}.txt"));
        std::fs::write(&path, redact(&r)).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Load the untouched source stage's route, tile cuts and candidate pieces.
    fn list_segments(&mut self) {
        let p = self.prefs.clone();
        self.log
            .push_str(&format!("\n=== stock-stage sections of {} ===\n", p.stage));
        self.tab = Tab::Map;
        self.show_stock = true;
        self.source_preview = None;
        self.selected_source_piece = None;
        self.section_filter.clear();
        self.sequence_note = None;
        let paks = PathBuf::from(&p.acr_dir)
            .join(r"acr\Content\Paks")
            .display()
            .to_string();
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend([
                    "segments".to_string(),
                    "--level".into(),
                    p.stage.clone(),
                    "--paks".into(),
                    paks,
                    "--json".into(),
                ]);
                let (tx, rx) = channel();
                self.job = Some(Job {
                    rx,
                    remaining: vec![],
                    total: 1,
                    progress: 0.05,
                    phase: "extracting stock-stage sections".into(),
                    kind: JobKind::ListPieces,
                });
                spawn_build(prog, cwd, full, None, tx);
            }
            Err(e) => self.log.push_str(&format!("backend: {e}\n")),
        }
    }

    /// Append the selected stock piece while preserving endpoint continuity.
    fn add_selected_piece(&mut self) {
        let Some(source) = self.source_preview.as_ref() else {
            return;
        };
        let Some(id) = self.selected_source_piece.as_deref() else {
            return;
        };
        let Some(piece) = source.pieces.iter().find(|p| p.id == id) else {
            return;
        };
        match append_piece_to_sequence(
            &self.prefs.sequence,
            piece,
            source.cuts.len().saturating_sub(1),
        ) {
            Ok(next) => {
                self.prefs.sequence = next;
                self.prefs.use_sequence = true;
                self.sequence_note = Some(format!("Added {}. Sequence endpoints remain connected; build validation will check join geometry.", piece.id));
            }
            Err(e) => self.sequence_note = Some(e),
        }
    }

    /// Remove one installed generated stage (by level name) or all of them.
    fn uninstall(&mut self, stage: Option<String>) {
        let p = self.prefs.clone();
        self.log.push_str(&format!(
            "\n=== uninstall {} ===\n",
            stage.as_deref().unwrap_or("all")
        ));
        self.tab = Tab::Log;
        let paks = PathBuf::from(&p.acr_dir)
            .join(r"acr\Content\Paks")
            .display()
            .to_string();
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(["uninstall".to_string(), "--paks".into(), paks]);
                if let Some(s) = stage {
                    full.extend(["--stage".to_string(), s]);
                }
                let (tx, rx) = channel();
                self.job = Some(Job {
                    rx,
                    remaining: vec![],
                    total: 1,
                    progress: 0.5,
                    phase: "uninstalling".into(),
                    kind: JobKind::Uninstall,
                });
                spawn_build(prog, cwd, full, None, tx);
            }
            Err(e) => self.log.push_str(&format!("backend: {e}\n")),
        }
    }

    fn poll(&mut self) {
        let mut finished: Option<(bool, Option<PathBuf>, String, JobKind)> = None;
        if let Some(job) = &mut self.job {
            while let Ok(m) = job.rx.try_recv() {
                match m {
                    Msg::Line { line, stdout } => {
                        if let Some(rest) = line.strip_prefix("@progress ") {
                            let (f, phase) = rest.split_once(' ').unwrap_or((rest, ""));
                            job.progress = f.trim().parse().unwrap_or(job.progress);
                            job.phase = phase.to_string();
                            continue;
                        }
                        // `segments --json` can be a large single line. Keep it in
                        // the result payload rather than flooding the visible log.
                        if !(stdout && job.kind == JobKind::ListPieces) {
                            self.log.push_str(&line);
                            self.log.push('\n');
                        }
                    }
                    Msg::Done { ok, out, stdout } => {
                        finished = Some((ok, out, stdout, job.kind));
                    }
                }
            }
        }
        if let Some((ok, out, stdout, kind)) = finished {
            match kind {
                JobKind::Build => {
                    if ok {
                        if let Some(out) = out {
                            match load_preview(&out) {
                                Ok(pv) => {
                                    self.preview = Some(pv);
                                    self.show_stock = false;
                                    self.tab = Tab::Map;
                                }
                                Err(e) => self.log.push_str(&format!("preview: {e}\n")),
                            }
                        }
                        self.next();
                    } else {
                        self.log.push_str("build failed (see above)\n");
                        self.job = None;
                    }
                    if self.job.is_none() {
                        self.sync_firewall();
                    }
                }
                JobKind::ListPieces => {
                    self.job = None;
                    if ok {
                        match parse_source_preview(&stdout) {
                            Ok(source) if source.level == self.prefs.stage
                                && !source.route_xyz.is_empty()
                                && source.route_xyz.len() == source.route_s.len()
                                && source.cuts.len() >= 2
                                && source.sections.len() == source.cuts.len() - 1 =>
                            {
                                let first = source.pieces.iter().find(|p| p.a == 0)
                                    .or_else(|| source.pieces.first()).map(|p| p.id.clone());
                                self.selected_source_piece = first;
                                self.source_preview = Some(source);
                                self.show_stock = true;
                                self.tab = Tab::Map;
                                if let Some(src) = &self.source_preview {
                                    self.log.push_str(&format!(
                                        "loaded stock map: {} cut-to-cut sections, {} buildable manual pieces; click a coloured section or choose a longer candidate from the list\n",
                                        src.sections.len(), src.pieces.len()));
                                }
                            }
                            Ok(source) => {
                                self.log.push_str(&format!(
                                    "stock map data was incomplete for {} (route points: {}, cuts: {}, map sections: {}, candidates: {})\n",
                                    source.level, source.route_xyz.len(), source.cuts.len(), source.sections.len(), source.pieces.len()));
                            }
                            Err(e) => self.log.push_str(&format!(
                                "could not read stock-stage map data: {e}. Rebuild the GUI with the updated StageGen backend.\n")),
                        }
                    } else {
                        self.log.push_str(
                            "could not extract stock-stage sections (see errors above)\n",
                        );
                    }
                }
                JobKind::Uninstall => {
                    self.job = None;
                    if !ok {
                        self.log.push_str("uninstall failed (see above)\n");
                    }
                    self.sync_firewall();
                }
            }
        }
    }
}

/// (program, leading args, working dir) of the backend: the embedded exe, or
/// `python -m stagegen.cli` in development.
fn backend_cmd(p: &Prefs) -> Result<(String, Vec<String>, String), String> {
    if p.use_python || BACKEND.is_none() {
        if !Path::new(&p.backend_dir).join("stagegen").is_dir() {
            return Err(format!("no stagegen package in {:?}", p.backend_dir));
        }
        return Ok((
            p.python.clone(),
            vec!["-m".into(), "stagegen.cli".into()],
            p.backend_dir.clone(),
        ));
    }
    let exe = embedded_backend()?;
    let cwd = exe
        .parent()
        .map(|d| d.display().to_string())
        .unwrap_or_default();
    Ok((exe.display().to_string(), vec![], cwd))
}

/// Forward subprocess output one line at a time without assuming that every
/// byte sequence is valid UTF-8. Unreal/package names and native libraries can
/// occasionally emit non-UTF-8 bytes on Windows. `BufRead::lines()` treats
/// that as a fatal read error; the loop then drops the pipe, and the backend
/// can fail on its next stdout write (for example, the final progress update).
fn forward_pipe_lines<R: std::io::Read>(reader: R, mut forward: impl FnMut(String)) {
    let mut reader = BufReader::new(reader);
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        match reader.read_until(b'\n', &mut bytes) {
            Ok(0) => break,
            Ok(_) => {
                if bytes.last() == Some(&b'\n') {
                    bytes.pop();
                }
                if bytes.last() == Some(&b'\r') {
                    bytes.pop();
                }
                forward(String::from_utf8_lossy(&bytes).into_owned());
            }
            Err(e) => {
                forward(format!("[subprocess output read error: {e}]"));
                break;
            }
        }
    }
}

fn spawn_build(
    python: String,
    cwd: String,
    args: Vec<String>,
    out: Option<PathBuf>,
    tx: Sender<Msg>,
) {
    std::thread::spawn(move || {
        use std::os::windows::process::CommandExt;
        let child = Command::new(&python)
            .args(&args)
            .current_dir(&cwd)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Msg::Line {
                    line: format!("cannot start {python}: {e}"),
                    stdout: false,
                });
                let _ = tx.send(Msg::Done {
                    ok: false,
                    out,
                    stdout: String::new(),
                });
                return;
            }
        };
        let err = child.stderr.take().unwrap();
        let tx2 = tx.clone();
        let t = std::thread::spawn(move || {
            forward_pipe_lines(err, |line| {
                let _ = tx2.send(Msg::Line {
                    line,
                    stdout: false,
                });
            });
        });
        let mut captured_stdout = String::new();
        forward_pipe_lines(child.stdout.take().unwrap(), |line| {
            captured_stdout.push_str(&line);
            captured_stdout.push('\n');
            let _ = tx.send(Msg::Line { line, stdout: true });
        });
        let _ = t.join();
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let _ = tx.send(Msg::Done {
            ok,
            out,
            stdout: captured_stdout,
        });
    });
}

fn path_row(ui: &mut egui::Ui, label: &str, value: &mut String, ok: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(value).desired_width(230.0));
        ui.label(if ok { "✔" } else { "✖" });
    });
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.prefs);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let busy = self.job.is_some();

        if !self.installed.is_empty() {
            let mut do_uninstall: Option<Option<String>> = None;
            let mut do_block = false;
            let list = self.installed.clone();
            egui::TopBottomPanel::top("installed").show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    let replaces = list.iter().any(|i| i.display.is_none());
                    ui.colored_label(egui::Color32::from_rgb(230, 150, 40),
                        if replaces {
                            format!("⚠ {} generated stage(s) installed (one replaces its original stage). Play offline.",
                                    list.len())
                        } else {
                            format!("{} generated stage(s) installed: extra stages in Free Practice under their rally \
                                     (list on the left). Play offline.", list.len())
                        });
                    if self.blocked {
                        ui.colored_label(egui::Color32::from_rgb(90, 180, 90), "🔒 ACR online blocked");
                    } else if !busy && ui.small_button("Block ACR online").clicked() {
                        do_block = true;
                    }
                    if !busy && ui.small_button("Uninstall all").clicked() {
                        do_uninstall = Some(None);
                    }
                });
                ui.add_space(4.0);
            });
            if do_block {
                self.prefs.block_online = true;
                self.sync_firewall();
            }
            if let Some(which) = do_uninstall {
                self.uninstall(which);
            }
        }

        let mut block_changed = false;
        let mut want_segments = false;
        let mut want_add_piece = false;
        let mut inst_action: Option<InstAction> = None;
        let pv_seq = self
            .preview
            .as_ref()
            .map(|pv| pv.sequence.clone())
            .filter(|s| !s.is_empty());
        let selected_piece_for_sidebar = self.source_preview.as_ref().and_then(|src| {
            self.selected_source_piece
                .as_deref()
                .and_then(|id| src.pieces.iter().find(|piece| piece.id == id))
                .cloned()
        });
        egui::SidePanel::left("opts").min_width(330.0).show(ctx, |ui| {
          egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("ACR Remixer");
            ui.add_space(6.0);
            egui::CollapsingHeader::new("Setup").default_open(self.prefs.acr_dir.is_empty()).show(ui, |ui| {
                let p = &mut self.prefs;
                let acr_ok = steam::is_acr_dir(Path::new(&p.acr_dir));
                path_row(ui, "ACR", &mut p.acr_dir, acr_ok);
                if ui.small_button("detect").clicked() {
                    if let Some(d) = steam::detect_acr() {
                        p.acr_dir = d.display().to_string();
                    }
                }
                path_row(ui, "Output", &mut p.out_dir, true);
                if BACKEND.is_some() {
                    ui.checkbox(&mut p.use_python, "Use Python backend (development)");
                }
                if p.use_python || BACKEND.is_none() {
                    let be_ok = Path::new(&p.backend_dir).join("stagegen").is_dir();
                    path_row(ui, "Backend", &mut p.backend_dir, be_ok);
                    path_row(ui, "Python", &mut p.python, true);
                }
            });
            ui.separator();
            if !self.prefs.code.trim().is_empty() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.strong("Generate builds a share code:");
                    ui.add(egui::Label::new(egui::RichText::new(&self.prefs.code).monospace().small()).wrap());
                    ui.horizontal(|ui| {
                        ui.small("The settings below are ignored while a code is loaded.");
                        if ui.add_enabled(!busy, egui::Button::new("✕ Use the form").small()).clicked() {
                            self.prefs.code.clear();
                        }
                    });
                });
            }
            ui.add_enabled_ui(!busy, |ui| {
                let p = &mut self.prefs;
                let cur = STAGES.iter().find(|s| s.0 == p.stage).map(|s| s.1).unwrap_or("?");
                egui::ComboBox::from_label("Stage").width(220.0).selected_text(cur).show_ui(ui, |ui| {
                    for (id, name) in STAGES {
                        let tag = if VERIFIED.contains(id) { "" } else { "  (experimental)" };
                        ui.selectable_value(&mut p.stage, id.to_string(), format!("{name}{tag}"));
                    }
                });
                if ui.button("List pieces / inspect stock map")
                    .on_hover_text("Load every source road section, its section IDs and the full elevation profile")
                    .clicked()
                {
                    want_segments = true;
                }
                ui.horizontal(|ui| {
                    ui.label("Length");
                    // open-ended: the source road (times Repeats) is the real limit
                    ui.add(
                        egui::DragValue::new(&mut p.length_m)
                            .range(1000.0..=f32::MAX)
                            .speed(100.0)
                            .suffix(" m"),
                    )
                    .on_hover_text(
                        "Type a length or drag. The build picks the closest length the \
                         source stage allows: about 90 % of its road times Repeats",
                    );
                });
                if p.length_m > 15000.0 {
                    ui.small(
                        "Long stage: may need Repeats above 1, takes longer to build and \
                         more memory (also in the game).",
                    );
                }
                ui.horizontal(|ui| {
                    ui.label("Seed");
                    ui.add(egui::DragValue::new(&mut p.seed).range(1..=999_999));
                    if ui.small_button("🎲").on_hover_text("random seed").clicked() {
                        p.seed = (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.subsec_nanos())
                            .unwrap_or(1)
                            % 999_999)
                            + 1;
                    }
                    ui.label("Batch");
                    ui.add(egui::DragValue::new(&mut p.batch).range(1..=50))
                        .on_hover_text("Build this many stages (seed, seed+1, ...)");
                });
                ui.add_enabled_ui(!p.use_sequence, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Repeats");
                        ui.add(egui::DragValue::new(&mut p.max_uses).range(1..=5))
                            .on_hover_text("How often a piece of road may appear. 1: each piece at most once \
                                            (length up to ~90 % of the original); 2-3: longer stages from \
                                            repeated pieces (each repeat costs some memory in the game)");
                    });
                });
                egui::CollapsingHeader::new("Sequence (choose the pieces yourself)")
                    .default_open(p.use_sequence)
                    .show(ui, |ui| {
                        ui.checkbox(&mut p.use_sequence, "Build this sequence instead of a seed")
                            .on_hover_text("e.g. 0-1 7-11 7-11r 7-11 18-19 to practise piece 7-11 three times. \
                                            Numbers are tile crossings (see 'List pieces'), r = backwards; \
                                            starts with the start piece 0-N, ends with the finish piece.");
                        ui.add(egui::TextEdit::multiline(&mut p.sequence).desired_rows(2).desired_width(300.0)
                            .hint_text("0-1 7-11 7-11r 7-11 18-19"));
                        ui.horizontal(|ui| {
                            if ui.add_enabled(pv_seq.is_some(), egui::Button::new("From the shown stage"))
                                .on_hover_text("Copy the pieces of the generated stage shown on the map, to tweak them")
                                .clicked()
                            {
                                p.sequence = pv_seq.clone().unwrap_or_default();
                                p.use_sequence = true;
                            }
                        });
                        if let Some(piece) = &selected_piece_for_sidebar {
                            ui.horizontal(|ui| {
                                ui.small(format!("Selected: {}  •  {:.0} m  •  {:.0}–{:.0} m elevation",
                                    piece.id, piece.length, piece.elevation_min, piece.elevation_max));
                                if ui.small_button("Add selected section").clicked() {
                                    want_add_piece = true;
                                }
                            });
                        }
                        if let Some(note) = &self.sequence_note {
                            ui.small(note);
                        }
                    });
                ui.label("Foliage (untick to leave out for performance):");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut p.forest, "Forest");
                    ui.checkbox(&mut p.grass, "Grass & rocks");
                    ui.checkbox(&mut p.foliage, "Ground grass");
                });
                ui.checkbox(&mut p.install, "Install into the game (closes ACR)");
                ui.add_enabled(p.install, egui::Checkbox::new(&mut p.launch, "Start ACR afterwards"));
                block_changed = ui.checkbox(&mut p.block_online, "Block ACR online while a stage is installed")
                    .on_hover_text("Adds a Windows Firewall rule for acr.exe while a generated stage is installed and \
                                    removes it on uninstall, so no times reach the online leaderboards. \
                                    Windows asks for admin rights each time it changes.")
                    .changed();
            });
            ui.add_space(8.0);
            let ready = steam::is_acr_dir(Path::new(&self.prefs.acr_dir))
                && (BACKEND.is_some() && !self.prefs.use_python
                    || Path::new(&self.prefs.backend_dir).join("stagegen").is_dir());
            ui.horizontal(|ui| {
                if let Some(job) = &self.job {
                    ui.vertical(|ui| {
                        let k = job.total - job.remaining.len();
                        let prefix = if job.total > 1 { format!("stage {k} of {}: ", job.total) } else { String::new() };
                        ui.add(egui::ProgressBar::new(job.progress).desired_width(300.0).animate(true)
                            .text(format!("{prefix}{} ({:.0} %)", job.phase, job.progress * 100.0)));
                    });
                } else if ui.add_enabled(ready, egui::Button::new("▶  Generate").min_size([140.0, 30.0].into())).clicked() {
                    self.start();
                }
                if !busy && ui.button("Open last").clicked() {
                    match load_preview(&self.out_for(self.prefs.seed)) {
                        Ok(pv) => {
                            self.preview = Some(pv);
                            self.show_stock = false;
                        }
                        Err(e) => self.log.push_str(&format!("preview: {e}\n")),
                    }
                }
            });
            if !self.installed.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new(format!("Installed stages ({})", self.installed.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        let list = self.installed.clone();
                        for inst in &list {
                            let label = inst.display.clone()
                                .unwrap_or_else(|| format!("{} (replaces the original)", inst.name));
                            let shown = self.shown_installed.as_deref() == Some(inst.name.as_str());
                            let old = "built with an older version: rebuild it from the form";
                            ui.horizontal(|ui| {
                                if ui.add_enabled(inst.layout.is_some(), egui::SelectableLabel::new(shown, &label))
                                    .on_hover_text("Show its map").on_disabled_hover_text("no map stored (older build)")
                                    .clicked()
                                {
                                    inst_action = Some(InstAction::Show(inst.clone()));
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.add_space(12.0);
                                let has = inst.params.is_some();
                                if ui.add_enabled(has, egui::Button::new("✎ Load").small())
                                    .on_hover_text("Put its settings into the form, to vary them")
                                    .on_disabled_hover_text(old).clicked()
                                {
                                    inst_action = Some(InstAction::Load(inst.clone()));
                                }
                                if ui.add_enabled(has, egui::Button::new("📋 Share").small())
                                    .on_hover_text("Copy its share code: anyone with ACR Remixer gets the same stage")
                                    .on_disabled_hover_text(old).clicked()
                                {
                                    inst_action = Some(InstAction::Share(inst.clone()));
                                }
                                if ui.add_enabled(has && !busy && ready, egui::Button::new("⟳ Regenerate").small())
                                    .on_hover_text("Build it again with the same settings and reinstall it \
                                                    (e.g. after an update with fixes; closes ACR)")
                                    .on_disabled_hover_text(old).clicked()
                                {
                                    inst_action = Some(InstAction::Regenerate(inst.clone()));
                                }
                                if ui.add_enabled(!busy && ready, egui::Button::new("🗑 Uninstall").small())
                                    .on_hover_text("Remove it from the game (closes ACR)").clicked()
                                {
                                    inst_action = Some(InstAction::Uninstall(inst.name.clone()));
                                }
                            });
                        }
                    });
            }
            if !VERIFIED.contains(&self.prefs.stage.as_str()) {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "Experimental stage: not yet verified in-game.");
            }
            if let Some(pv) = &self.preview {
                ui.separator();
                ui.label(format!("{}: {:.2} km, {} pieces, {} pacenotes", pv.name, pv.length / 1000.0,
                                 pv.joins.len() + 1, pv.notes));
            }
            ui.separator();
            ui.separator();
            ui.strong("🔗 Share a stage");
            ui.label("A code is all a friend needs: it rebuilds exactly the stage on the map, on their PC.");
            match self.preview.as_ref().map(|pv| pv.share_code.clone()).filter(|c| !c.is_empty()) {
                Some(code) => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Stage on the map:");
                        ui.monospace(&code);
                        if ui.small_button("📋 Copy").on_hover_text("Copy this stage's share code").clicked() {
                            ui.ctx().copy_text(code.clone());
                            self.share_note = Some(format!("Copied: {code}"));
                        }
                    });
                }
                None => {
                    ui.small("Generate a stage (or pick an installed one) to get its code.");
                }
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.code_input).hint_text("paste a code from a friend")
                    .desired_width(220.0));
                if ui.add_enabled(!busy, egui::Button::new("Load")).on_hover_text("Use this code for the next \
                    Generate").clicked()
                {
                    let input = norm_code(&self.code_input);
                    if let Some(level) = code_level(&input) {
                        if STAGES.iter().any(|s| s.0 == level) {
                            self.prefs.stage = level;
                            self.prefs.code = input;
                            self.share_note = Some("Code loaded: press Generate to build that stage.".into());
                        } else {
                            self.share_note = Some(format!("This code is for a stage not in this version: {level}"));
                        }
                    } else if input.contains('/') && apply_share_code(&mut self.prefs, &input).is_ok() {
                        self.share_note = Some("An old code (ACR Remixer 0.1.x): its settings are loaded, but this \
                            version finds a different stage with them. Ask for a new code.".into());
                    } else {
                        self.share_note = Some("Not a share code (they start with R2:).".into());
                    }
                }
            });
            if let Some(n) = &self.share_note {
                ui.small(n);
            }
            ui.small("Installed stages have their own 📋 Share button in the list above.");
            ui.add_space(4.0);
            egui::CollapsingHeader::new("Bug report").show(ui, |ui| {
                if ui.button("Export bug report").on_hover_text(
                    "Writes one text file (settings, last log, ACR crash reports; your user name is hidden) \
                     to attach to a bug report").clicked()
                {
                    match self.export_bug_report() {
                        Ok(path) => {
                            self.log.push_str(&format!("bug report written: {}\n", redact(&path.display().to_string())));
                            let _ = Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
                        }
                        Err(e) => self.log.push_str(&format!("bug report: {e}\n")),
                    }
                }
                ui.small(format!("ACR Remixer {VERSION}"));
            });
            ui.add_space(6.0);
            ui.small(format!("ACR Remixer {VERSION} · © 2026 hiro98 (vale981) and contributors"));
            ui.horizontal(|ui| {
                ui.small("Free software under the GNU GPL v3 or later, without any warranty.");
                ui.hyperlink_to(egui::RichText::new("License").small(),
                                "https://github.com/vale981/acr-remixer/blob/main/LICENSE");
            });
          });
        });

        if block_changed && !busy {
            self.sync_firewall();
        }
        if want_segments && !busy {
            self.list_segments();
        }
        match inst_action {
            Some(InstAction::Show(i)) => self.show_installed(&i),
            Some(InstAction::Load(i)) => {
                if let Some(params) = &i.params {
                    self.prefs = prefs_with_params(&self.prefs, params);
                    self.log.push_str(&format!(
                        "settings of {} loaded into the form\n",
                        i.display.as_deref().unwrap_or(&i.name)
                    ));
                }
                self.show_installed(&i);
            }
            Some(InstAction::Share(i)) => {
                match i
                    .params
                    .as_ref()
                    .and_then(|p| p["code"].as_str())
                    .filter(|c| !c.is_empty())
                {
                    Some(code) => {
                        ctx.copy_text(code.to_string());
                        self.log.push_str(&format!("share code copied: {code}\n"));
                        self.share_note = Some(format!("Copied: {code}"));
                    }
                    None => {
                        self.share_note = Some(
                            "This stage was built before share codes existed: \
                                                ⟳ Regenerate it to get one."
                                .into(),
                        )
                    }
                }
            }
            Some(InstAction::Regenerate(i)) => self.regenerate(&i),
            Some(InstAction::Uninstall(name)) => {
                if self.shown_installed.as_deref() == Some(name.as_str()) {
                    self.shown_installed = None;
                }
                self.uninstall(Some(name));
            }
            None => {}
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Map, "Map");
                ui.selectable_value(&mut self.tab, Tab::Elevation, "Elevation");
                ui.selectable_value(&mut self.tab, Tab::Log, "Log");
                let has_current_stock = self.source_preview.as_ref()
                    .is_some_and(|source| source.level == self.prefs.stage);
                if has_current_stock {
                    ui.separator();
                    ui.selectable_value(&mut self.show_stock, true, "Stock stage");
                    if self.preview.is_some() {
                        ui.selectable_value(&mut self.show_stock, false, "Generated stage");
                    }
                }
            });
            ui.separator();
            match self.tab {
                Tab::Log => {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(&self.log).monospace()).wrap(),
                            );
                        });
                }
                Tab::Map if self.show_stock && self.source_preview.as_ref()
                    .is_some_and(|source| source.level == self.prefs.stage) => {
                    let source = self.source_preview.as_ref().unwrap();
                    let mut selected = self.selected_source_piece.clone();
                    // map on top, a flat height profile in the same colours below,
                    // the selection and the list of candidates under them
                    let elev_h = 130.0;
                    let map_h = (ui.available_height() - elev_h - 300.0).max(240.0);
                    if let Some(id) = source_map_ui(ui, source, selected.as_deref(), map_h) {
                        selected = Some(id);
                    }
                    if let Some(id) = source_elevation_ui(ui, source, selected.as_deref(), elev_h) {
                        selected = Some(id);
                    }
                    if let Some(piece) = selected.as_deref().and_then(|id| find_source_piece(source, id)) {
                        let can_add = source.pieces.iter().any(|p| p.id == piece.id);
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(format!("Selected section {}", piece.id));
                            ui.label(format!("{:.0} m", piece.length));
                            ui.label(format!("start {:.1} m → end {:.1} m", piece.elevation_start, piece.elevation_end));
                            ui.label(format!("range {:.1}–{:.1} m", piece.elevation_min, piece.elevation_max));
                            ui.label(if piece.reversible { "reversible" } else { "forward only" });
                            if !can_add {
                                ui.colored_label(egui::Color32::from_gray(150), "inspect only: not a standalone build candidate");
                            }
                            if ui.add_enabled(can_add, egui::Button::new("Add to manual sequence")).clicked() {
                                want_add_piece = true;
                            }
                        });
                    }
                    if let Some(id) = source_sections_ui(ui, source, selected.as_deref(), &mut self.section_filter) {
                        selected = Some(id);
                    }
                    self.selected_source_piece = selected;
                }
                Tab::Elevation if self.show_stock && self.source_preview.as_ref()
                    .is_some_and(|source| source.level == self.prefs.stage) => {
                    let source = self.source_preview.as_ref().unwrap();
                    let mut selected = self.selected_source_piece.clone();
                    if let Some(id) = source_elevation_tab_ui(ui, source, selected.as_deref()) {
                        selected = Some(id);
                    }
                    if let Some(piece) = selected.as_deref().and_then(|id| find_source_piece(source, id)) {
                        let status = if piece.buildable { "buildable" } else { "inspect only" };
                        ui.label(format!("Section {} ({status}): {:.0} m; {:.1} → {:.1} m; min/max {:.1}/{:.1} m",
                            piece.id, piece.length, piece.elevation_start, piece.elevation_end,
                            piece.elevation_min, piece.elevation_max));
                    }
                    if let Some(id) = source_sections_ui(ui, source, selected.as_deref(), &mut self.section_filter) {
                        selected = Some(id);
                    }
                    self.selected_source_piece = selected;
                }
                Tab::Map if self.preview.is_some() => {
                    let pv = self.preview.as_ref().unwrap();
                    // map on top, a flat height profile in the same colours below
                    let elev_h = 130.0;
                    let map_h = (ui.available_height() - elev_h - 10.0).max(240.0);
                    Plot::new("map")
                        .data_aspect(1.0)
                        .legend(Legend::default().follow_insertion_order(true))
                        .height(map_h)
                        .show(ui, |pu| {
                            for (k, pts) in piece_spans(pv).into_iter().enumerate() {
                                let pts: Vec<[f64; 2]> = pts.iter().map(|(p, _)| [p[0], p[1]]).collect();
                                pu.line(
                                    Line::new(PlotPoints::from(pts))
                                        .width(3.0_f32)
                                        .color(section_color(k, true))
                                        .name(piece_label(pv, k)),
                                );
                            }
                            if let (Some(a), Some(b)) = (pv.pts.first(), pv.pts.last()) {
                                pu.points(
                                    egui_plot::Points::new(vec![[a[0], a[1]]])
                                        .radius(6.0_f32)
                                        .name("start"),
                                );
                                pu.points(
                                    egui_plot::Points::new(vec![[b[0], b[1]]])
                                        .radius(6.0_f32)
                                        .shape(egui_plot::MarkerShape::Square)
                                        .name("finish"),
                                );
                            }
                        });
                    preview_elevation_ui(ui, pv, elev_h);
                }
                Tab::Elevation if self.preview.is_some() => {
                    let pv = self.preview.as_ref().unwrap();
                    preview_elevation_ui(ui, pv, ui.available_height());
                }
                Tab::Map | Tab::Elevation if self.show_stock => {
                    ui.centered_and_justified(|ui| {
                        if let Some(source) = &self.source_preview {
                            ui.label(format!("The loaded stock map is {}. Select that stage or click List pieces again to load {}.",
                                source.level, self.prefs.stage));
                        } else if self.job.is_some() {
                            ui.label("Extracting the stock road network, tile sections and elevations…");
                        } else {
                            ui.label("Select a stock stage and click List pieces to inspect its complete road map, sections and elevations.");
                        }
                    });
                }
                _ => {
                    ui.centered_and_justified(|ui| ui.label("Generate a stage or select List pieces to inspect the original stock route."));
                }
            }
        });
        if want_add_piece && !busy {
            self.add_selected_piece();
        }
    }
}

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("ACR Remixer"),
        ..Default::default()
    };
    // the app id stays "acr-stagegen": eframe stores the saved settings under it
    eframe::run_native(
        "acr-stagegen",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        append_piece_to_sequence, build_dir, code_level, forward_pipe_lines, norm_code,
        parse_source_preview, Prefs, SourcePiece,
    };

    const CODE: &str = "R2:WalesS4HafrenSouth:0-1352,2451-3255,3255-4300,4656-5592:FG-:5751:EU";

    #[test]
    fn share_code_level_and_whitespace() {
        assert_eq!(code_level(CODE).as_deref(), Some("WalesS4HafrenSouth"));
        assert_eq!(
            code_level(&format!(" {}\n{} ", &CODE[..30], &CODE[30..])).as_deref(),
            Some("WalesS4HafrenSouth")
        );
        assert_eq!(
            norm_code(&format!("{}\n {}", &CODE[..30], &CODE[30..])),
            CODE
        );
        assert_eq!(code_level("WalesS4HafrenSouth/4500/1/FGL"), None);
        assert_eq!(code_level("R2::x"), None);
    }

    #[test]
    fn code_build_folder_matches_the_backend() {
        // the backend (stagegen/cli.py, _fnv_tag) names this code's folder ..._code7EF3A987
        let p = Prefs {
            stage: "WalesS4HafrenSouth".into(),
            out_dir: "out".into(),
            code: CODE.into(),
            ..Default::default()
        };
        assert!(build_dir(&p, 1).ends_with("WalesS4HafrenSouth_code7EF3A987"));
    }

    fn lines(input: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        forward_pipe_lines(input, |l| out.push(l));
        out
    }

    #[test]
    fn plain_and_crlf_lines() {
        assert_eq!(lines(b"one\ntwo\r\nthree"), ["one", "two", "three"]);
        assert!(lines(b"").is_empty());
    }

    #[test]
    fn invalid_utf8_does_not_stop_reading() {
        // a Latin-1 byte (as in some package names) in the middle of the output
        let got = lines(b"@progress 0.5 building\npath caf\xe9\n@progress 1.0 done\n");
        assert_eq!(got.len(), 3);
        assert!(got[1].starts_with("path caf"));
        assert_eq!(got[2], "@progress 1.0 done");
    }

    fn piece(a: usize, b: usize, reversible: bool) -> SourcePiece {
        SourcePiece {
            id: format!("{a}-{b}"),
            a,
            b,
            length: 100.0,
            reversible,
            buildable: true,
            station_start: 0.0,
            station_end: 100.0,
            station: vec![],
            midpoint: [0.0; 3],
            elevation_start: 0.0,
            elevation_end: 0.0,
            elevation_min: 0.0,
            elevation_max: 0.0,
            xyz: vec![],
        }
    }

    #[test]
    fn adding_pieces_keeps_the_sequence_connected() {
        // starts at cut 0
        assert_eq!(
            append_piece_to_sequence("", &piece(0, 3, false), 9).unwrap(),
            "0-3"
        );
        assert!(append_piece_to_sequence("", &piece(3, 5, true), 9).is_err());
        // forwards from the exit, or backwards when the piece ends there
        assert_eq!(
            append_piece_to_sequence("0-3", &piece(3, 5, true), 9).unwrap(),
            "0-3 3-5"
        );
        assert_eq!(
            append_piece_to_sequence("0-3 3-5", &piece(3, 5, true), 9).unwrap(),
            "0-3 3-5 3-5r"
        );
        assert_eq!(
            append_piece_to_sequence("0-3 3-5 3-5r", &piece(3, 9, false), 9).unwrap(),
            "0-3 3-5 3-5r 3-9"
        );
        // a forward-only piece can't be driven backwards, a far piece doesn't connect
        assert!(append_piece_to_sequence("0-3 3-5", &piece(3, 5, false), 9).is_err());
        assert!(append_piece_to_sequence("0-3", &piece(4, 5, true), 9).is_err());
        // nothing after the finish, nothing after a broken chain
        assert!(append_piece_to_sequence("0-3 3-9", &piece(3, 5, true), 9).is_err());
        assert!(append_piece_to_sequence("0-3 4-5", &piece(5, 6, true), 9).is_err());
    }

    #[test]
    fn stock_preview_is_the_last_json_line() {
        let out =
            "some diagnostics\n{\"level\":\"L\",\"route_xyz\":[[0,0,1],[1,0,2]],\"route_s\":[0,1],\
                   \"cuts\":[{\"i\":0,\"s\":0,\"kind\":\"start\",\"x\":0,\"y\":0,\"z\":1}],\
                   \"sections\":[],\"pieces\":[]}\n";
        let src = parse_source_preview(out).unwrap();
        assert_eq!(src.level, "L");
        assert_eq!(src.route_xyz.len(), 2);
        assert!(parse_source_preview("no json here").is_err());
    }
}
