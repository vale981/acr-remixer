// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-FileCopyrightText: 2026 iceone88
// SPDX-License-Identifier: MIT

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
use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};
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
const BACKEND: Option<&[u8]> = Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../packaging/dist/stagegen-backend.exe")));
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
    let dir = PathBuf::from(std::env::var("LOCALAPPDATA").map_err(|e| e.to_string())?).join("acr-stagegen");
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
        let acr = steam::detect_acr().map(|p| p.display().to_string()).unwrap_or_default();
        let py = std::env::var("LOCALAPPDATA")
            .map(|l| PathBuf::from(l).join(r"Programs\Python\Python314\python.exe"))
            .ok()
            .filter(|p| p.is_file())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "python".into());
        let backend = if acr.is_empty() { String::new() } else { format!(r"{acr}\acr-stagegen") };
        let out = if backend.is_empty() { String::new() } else { format!(r"{backend}\out") };
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
    if let Some(s) = params["level"].as_str() { p.stage = s.to_string(); }
    if let Some(v) = params["length"].as_f64() { p.length_m = v as f32; }
    if let Some(v) = params["seed"].as_u64() { p.seed = v as u32; }
    p.max_uses = params["max_uses"].as_u64().unwrap_or(1) as u32;
    p.sequence = params["sequence"].as_str().unwrap_or("").to_string();
    p.use_sequence = !p.sequence.trim().is_empty();
    p.forest = params["forest"].as_bool().unwrap_or(true);
    p.grass = params["grass"].as_bool().unwrap_or(true);
    p.foliage = params["foliage"].as_bool().unwrap_or(true);
    p
}

/// Generated stages installed in ~mods (the shared menu-table container excluded).
fn installed_stages(acr_dir: &str) -> Vec<Installed> {
    let mods = PathBuf::from(acr_dir).join(r"acr\Content\Paks\~mods");
    let Ok(rd) = std::fs::read_dir(&mods) else { return vec![] };
    let mut v: Vec<Installed> = rd.filter_map(|e| e.ok()).filter_map(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        let name = n.strip_prefix("pakchunk99-")?.strip_suffix("_Stagegen_9_P.utoc")?.to_string();
        if name == "Tables" { return None; }
        let meta = mods.join(format!("pakchunk99-{name}_Stagegen_9_P.stagegen.json"));
        let j = std::fs::read_to_string(meta).ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let display = j.as_ref().and_then(|j| j["display"].as_str().map(|s| s.to_string()));
        let params = j.as_ref().map(|j| j["params"].clone()).filter(|p| p.is_object());
        let layout = Some(mods.join(format!("pakchunk99-{name}_Stagegen_9_P.layout.json"))).filter(|p| p.is_file());
        Some(Installed { name, display, params, layout })
    }).collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// Sequence text as the backend normalizes it (single spaces, no commas).
fn norm_sequence(s: &str) -> String {
    s.replace(',', " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tag of a piece sequence: FNV-1a 32-bit of the normalized text (= backend `_seq_tag`).
fn seq_tag(s: &str) -> String {
    let mut h: u32 = 0x811C_9DC5;
    for b in norm_sequence(s).bytes() {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    format!("{h:08X}")
}

/// Output folder of a build (matches the backend's naming).
fn build_dir(p: &Prefs, seed: u32) -> PathBuf {
    let base = PathBuf::from(&p.out_dir);
    if p.use_sequence && !p.sequence.trim().is_empty() {
        return base.join(format!("{}_seq{}", p.stage, seq_tag(&p.sequence)));
    }
    let uses = if p.max_uses > 1 { format!("_x{}", p.max_uses) } else { String::new() };
    base.join(format!("{}_seed{seed}_{:.0}m{uses}", p.stage, p.length_m))
}

const FW_RULE: &str = "acr-stagegen: block ACR online";

fn quiet(cmd: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000) // CREATE_NO_WINDOW
}

/// Is the firewall rule that blocks ACR's network access present?
fn fw_blocked() -> bool {
    quiet(Command::new("netsh").args(["advfirewall", "firewall", "show", "rule", &format!("name={FW_RULE}")]))
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
        .iter().map(|(on, c)| if *on { *c } else { '-' }).collect();
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
        return Err("expected Stage/Length/Seed/Foliage, e.g. WalesS4HafrenSouth/4500/1/FGL".into());
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
    p.length_m = len.clamp(1000.0, 15000.0);
    p.seed = seed.max(1);
    p.forest = parts[3].contains('F');
    p.grass = parts[3].contains('G');
    p.foliage = parts[3].contains('L');
    Ok(())
}

/// Summary of the newest ACR crash reports (error + top of the call stack).
fn acr_crashes(n: usize) -> String {
    let Ok(base) = std::env::var("LOCALAPPDATA") else { return "(no LOCALAPPDATA)\n".into() };
    let dir = PathBuf::from(base).join(r"acr\Saved\Crashes");
    let Ok(rd) = std::fs::read_dir(&dir) else { return "(no crash reports)\n".into() };
    let mut ds: Vec<_> = rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
    ds.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = String::new();
    for (t, d) in ds.into_iter().take(n) {
        let age = t.elapsed().map(|e| e.as_secs() / 60).unwrap_or(0);
        let txt = std::fs::read_to_string(d.join("CrashContext.runtime-xml")).unwrap_or_default();
        let tag = |name: &str| -> String {
            let (a, b) = (format!("<{name}>"), format!("</{name}>"));
            txt.find(&a).and_then(|i| txt[i + a.len()..].find(&b).map(|j| txt[i + a.len()..i + a.len() + j].trim().to_string()))
                .unwrap_or_default()
        };
        let stack: Vec<String> = tag("PCallStack").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).take(15).collect();
        out.push_str(&format!("--- crash {} min ago: {}\n{}\n", age, tag("ErrorMessage"), stack.join("\n")));
    }
    if out.is_empty() { "(no crash reports)\n".into() } else { out }
}

fn windows_version() -> String {
    quiet(Command::new("cmd").args(["/c", "ver"])).output()
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
}

fn load_preview(dir: &Path) -> Result<Preview, String> {
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
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
            Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
        })
        .collect();
    let mut s = vec![0.0];
    for w in pts.windows(2) {
        let d = ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt();
        s.push(s.last().unwrap() + d);
    }
    let joins = v["pieces"]
        .as_array()
        .map(|a| a.iter().skip(1).filter_map(|p| p["s_start"].as_f64()).collect())
        .unwrap_or_default();
    let sequence = v["pieces"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let id = p["segment"].as_str()?.rsplit('#').next()?.to_string();
                    Some(if p["reversed"].as_bool().unwrap_or(false) { id + "r" } else { id })
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
    })
}

enum Msg {
    Line(String),
    Done { ok: bool, out: Option<PathBuf> },
}

struct Job {
    rx: Receiver<Msg>,
    remaining: Vec<u32>,
    total: usize,
    progress: f32,
    phase: String,
}

struct App {
    prefs: Prefs,
    log: String,
    job: Option<Job>,
    preview: Option<Preview>,
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
        let prefs: Prefs = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)).unwrap_or_default();
        let mut app = App { prefs, log: String::new(), job: None, preview: None, tab: Tab::Map,
                            installed: vec![], blocked: false, code_input: String::new(), job_prefs: None,
                            shown_installed: None, share_note: None };
        app.preview = load_preview(&app.out_for(app.prefs.seed)).ok();     // reopen the last stage
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
        let n = if self.using_sequence() { 1 } else { self.prefs.batch.max(1) };   // a sequence is one stage
        let seeds: Vec<u32> = (0..n).map(|i| self.prefs.seed + i).collect();
        self.log.clear();
        self.tab = Tab::Log;
        let (_, rx) = channel();
        let total = seeds.len();
        self.job_prefs = None;
        self.job = Some(Job { rx, remaining: seeds, total, progress: 0.0, phase: "starting".into() });
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
        self.log.push_str(&format!("regenerating {}\n", inst.display.as_deref().unwrap_or(&inst.name)));
        self.tab = Tab::Log;
        let (_, rx) = channel();
        let seed = p.seed;
        self.job_prefs = Some(p);
        self.job = Some(Job { rx, remaining: vec![seed], total: 1, progress: 0.0, phase: "starting".into() });
        self.next();
    }

    /// Show an installed stage's map.
    fn show_installed(&mut self, inst: &Installed) {
        let Some(file) = &inst.layout else { return };
        match load_preview_file(file, inst.display.clone().unwrap_or_else(|| inst.name.clone())) {
            Ok(pv) => {
                self.preview = Some(pv);
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
        let Some(seed) = (if job.remaining.is_empty() { None } else { Some(job.remaining.remove(0)) }) else {
            self.job = None;
            return;
        };
        let last = job.remaining.is_empty();
        let p = self.job_prefs.clone().unwrap_or_else(|| self.prefs.clone());
        let mut args: Vec<String> = vec![
            "build".into(),
            "--level".into(), p.stage.clone(),
            "--length".into(), format!("{:.0}", p.length_m),
            "--seed".into(), seed.to_string(),
            "--out".into(), p.out_dir.clone(),
        ];
        if p.use_sequence && !p.sequence.trim().is_empty() {
            args.push("--sequence".into());
            args.push(norm_sequence(&p.sequence));
        } else if p.max_uses > 1 {
            args.push("--max-uses".into());
            args.push(p.max_uses.to_string());
        }
        if !p.forest { args.push("--no-forest".into()); }
        if !p.grass { args.push("--no-grass".into()); }
        if !p.foliage { args.push("--no-foliage".into()); }
        // every stage of a batch is installed (each gets its own menu entry); ACR starts after the last
        if p.install { args.push("--install".into()); }
        if last && p.install && p.launch { args.push("--launch".into()); }
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks");
        args.push("--paks".into());
        args.push(paks.display().to_string());
        let out = build_dir(&p, seed);
        let (tx, rx) = channel();
        job.rx = rx;
        job.progress = 0.0;
        job.phase = "starting".into();
        self.log.push_str(&format!("\n=== {} seed {} ===\n", p.stage, seed));
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
                Ok(()) => self.log.push_str(if want { "ACR's internet access is blocked while the generated stage is installed.\n" }
                                            else { "ACR's internet access is allowed again.\n" }),
                Err(e) => self.log.push_str(&format!("firewall: {e}\n")),
            }
            self.blocked = fw_blocked();
        }
    }

    /// Write one text file with everything useful for a bug report; returns its path.
    fn export_bug_report(&self) -> Result<PathBuf, String> {
        let p = &self.prefs;
        let mut r = String::new();
        r.push_str(&format!("ACR Remixer bug report\nversion: {VERSION}\nbackend: {}\nwindows: {}\n\n",
            if p.use_python || BACKEND.is_none() { "python (development)" } else { "embedded" }, windows_version()));
        r.push_str(&format!("share code: {}\n", share_code(p)));
        let inst: Vec<&str> = self.installed.iter().map(|i| i.name.as_str()).collect();
        r.push_str(&format!("installed stages: {}\nonline blocked: {}\n\n",
            if inst.is_empty() { "none".to_string() } else { inst.join(", ") }, self.blocked));
        r.push_str("=== settings ===\n");
        r.push_str(&serde_json::to_string_pretty(p).unwrap_or_default());
        r.push_str("\n\n=== last stage ===\n");
        match &self.preview {
            Some(pv) => r.push_str(&format!("{}: {:.2} km, {} pieces (joins at {:?} m), {} pacenotes\n",
                pv.name, pv.length / 1000.0, pv.joins.len() + 1,
                pv.joins.iter().map(|j| j.round()).collect::<Vec<_>>(), pv.notes)),
            None => r.push_str("(none loaded)\n"),
        }
        r.push_str("\n=== log of the last run ===\n");
        r.push_str(if self.log.is_empty() { "(empty)\n" } else { &self.log });
        r.push_str("\n=== ACR crash reports (newest first) ===\n");
        r.push_str(&acr_crashes(3));
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let dir = if Path::new(&p.out_dir).is_dir() { PathBuf::from(&p.out_dir) } else { std::env::temp_dir() };
        let path = dir.join(format!("acr-remixer-bugreport-{stamp}.txt"));
        std::fs::write(&path, redact(&r)).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Show the selected stage's pieces (for a sequence) in the log.
    fn list_segments(&mut self) {
        let p = self.prefs.clone();
        self.log.push_str(&format!("\n=== pieces of {} ===\n", p.stage));
        self.tab = Tab::Log;
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks").display().to_string();
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(["segments".to_string(), "--level".into(), p.stage.clone(), "--paks".into(), paks]);
                let (tx, rx) = channel();
                self.job = Some(Job { rx, remaining: vec![], total: 1, progress: 0.5, phase: "reading pieces".into() });
                spawn_build(prog, cwd, full, None, tx);
            }
            Err(e) => self.log.push_str(&format!("backend: {e}\n")),
        }
    }

    /// Remove one installed generated stage (by level name) or all of them.
    fn uninstall(&mut self, stage: Option<String>) {
        let p = self.prefs.clone();
        self.log.push_str(&format!("\n=== uninstall {} ===\n", stage.as_deref().unwrap_or("all")));
        self.tab = Tab::Log;
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks").display().to_string();
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(["uninstall".to_string(), "--paks".into(), paks]);
                if let Some(s) = stage {
                    full.extend(["--stage".to_string(), s]);
                }
                let (tx, rx) = channel();
                self.job = Some(Job { rx, remaining: vec![], total: 1, progress: 0.5, phase: "uninstalling".into() });
                spawn_build(prog, cwd, full, None, tx);
            }
            Err(e) => self.log.push_str(&format!("backend: {e}\n")),
        }
    }

    fn poll(&mut self) {
        let mut finished = None;
        if let Some(job) = &mut self.job {
            while let Ok(m) = job.rx.try_recv() {
                match m {
                    Msg::Line(l) => {
                        if let Some(rest) = l.strip_prefix("@progress ") {
                            let (f, phase) = rest.split_once(' ').unwrap_or((rest, ""));
                            job.progress = f.trim().parse().unwrap_or(job.progress);
                            job.phase = phase.to_string();
                            continue;
                        }
                        self.log.push_str(&l);
                        self.log.push('\n');
                    }
                    Msg::Done { ok, out } => finished = Some((ok, out)),
                }
            }
        }
        if let Some((ok, out)) = finished {
            if ok {
                if let Some(out) = out {
                    match load_preview(&out) {
                        Ok(pv) => {
                            self.preview = Some(pv);
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
    }
}

/// (program, leading args, working dir) of the backend: the embedded exe, or
/// `python -m stagegen.cli` in development.
fn backend_cmd(p: &Prefs) -> Result<(String, Vec<String>, String), String> {
    if p.use_python || BACKEND.is_none() {
        if !Path::new(&p.backend_dir).join("stagegen").is_dir() {
            return Err(format!("no stagegen package in {:?}", p.backend_dir));
        }
        return Ok((p.python.clone(), vec!["-m".into(), "stagegen.cli".into()], p.backend_dir.clone()));
    }
    let exe = embedded_backend()?;
    let cwd = exe.parent().map(|d| d.display().to_string()).unwrap_or_default();
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

fn spawn_build(python: String, cwd: String, args: Vec<String>, out: Option<PathBuf>, tx: Sender<Msg>) {
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
                let _ = tx.send(Msg::Line(format!("cannot start {python}: {e}")));
                let _ = tx.send(Msg::Done { ok: false, out });
                return;
            }
        };
        let err = child.stderr.take().unwrap();
        let tx2 = tx.clone();
        let t = std::thread::spawn(move || {
            forward_pipe_lines(err, |line| {
                let _ = tx2.send(Msg::Line(line));
            });
        });
        forward_pipe_lines(child.stdout.take().unwrap(), |line| {
            let _ = tx.send(Msg::Line(line));
        });
        let _ = t.join();
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let _ = tx.send(Msg::Done { ok, out });
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
        let mut inst_action: Option<InstAction> = None;
        let pv_seq = self.preview.as_ref().map(|pv| pv.sequence.clone()).filter(|s| !s.is_empty());
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
            ui.add_enabled_ui(!busy, |ui| {
                let p = &mut self.prefs;
                let cur = STAGES.iter().find(|s| s.0 == p.stage).map(|s| s.1).unwrap_or("?");
                egui::ComboBox::from_label("Stage").width(220.0).selected_text(cur).show_ui(ui, |ui| {
                    for (id, name) in STAGES {
                        let tag = if VERIFIED.contains(id) { "" } else { "  (experimental)" };
                        ui.selectable_value(&mut p.stage, id.to_string(), format!("{name}{tag}"));
                    }
                });
                ui.add(egui::Slider::new(&mut p.length_m, 1000.0..=15000.0).step_by(100.0).suffix(" m").text("Length"))
                    .on_hover_text("Rounded to the closest length the source stage allows");
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
                                .on_hover_text("Copy the pieces of the stage shown on the map, to tweak them")
                                .clicked()
                            {
                                p.sequence = pv_seq.clone().unwrap_or_default();
                                p.use_sequence = true;
                            }
                            if ui.button("List pieces").on_hover_text("Show the pieces of this stage in the Log tab")
                                .clicked()
                            {
                                want_segments = true;
                            }
                        });
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
                        Ok(pv) => self.preview = Some(pv),
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
            ui.label("A code is all a friend needs: same code = same stage on their PC.");
            let code = share_code(&self.prefs);
            ui.horizontal(|ui| {
                ui.label("This form:");
                ui.monospace(&code);
                if ui.small_button("📋 Copy").on_hover_text("Copy the code of the stage set up in the form").clicked() {
                    ui.ctx().copy_text(code.clone());
                    self.share_note = Some(format!("Copied: {code}"));
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.code_input).hint_text("paste a code from a friend")
                    .desired_width(220.0));
                if ui.add_enabled(!busy, egui::Button::new("Load")).on_hover_text("Fill the form from the code; \
                    then press Generate").clicked()
                {
                    let input = self.code_input.clone();
                    match apply_share_code(&mut self.prefs, &input) {
                        Ok(()) => self.share_note = Some(format!("Loaded {} - press Generate", input.trim())),
                        Err(e) => self.share_note = Some(format!("Not a valid code: {e}")),
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
                    self.log.push_str(&format!("settings of {} loaded into the form\n",
                                               i.display.as_deref().unwrap_or(&i.name)));
                }
                self.show_installed(&i);
            }
            Some(InstAction::Share(i)) => {
                if let Some(params) = &i.params {
                    let code = share_code(&prefs_with_params(&self.prefs, params));
                    ctx.copy_text(code.clone());
                    self.log.push_str(&format!("share code copied: {code}\n"));
                    self.share_note = Some(format!("Copied: {code}"));
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
            });
            ui.separator();
            match self.tab {
                Tab::Log => {
                    egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(&self.log).monospace()).wrap());
                    });
                }
                Tab::Map | Tab::Elevation if self.preview.is_none() => {
                    ui.centered_and_justified(|ui| ui.label("Generate a stage (or open the last one) to see it here."));
                }
                Tab::Map => {
                    let pv = self.preview.as_ref().unwrap();
                    Plot::new("map").data_aspect(1.0).legend(Legend::default()).show(ui, |pu| {
                        let mut bounds = vec![0.0];
                        bounds.extend(pv.joins.iter().copied());
                        bounds.push(f64::INFINITY);
                        for (k, w) in bounds.windows(2).enumerate() {
                            let pts: Vec<[f64; 2]> = pv.pts.iter().zip(&pv.s)
                                .filter(|(_, s)| **s >= w[0] - 10.0 && **s <= w[1] + 10.0)
                                .map(|(p, _)| [p[0], p[1]]).collect();
                            pu.line(Line::new(PlotPoints::from(pts)).width(3.0_f32).name(format!("piece {k}")));
                        }
                        if let (Some(a), Some(b)) = (pv.pts.first(), pv.pts.last()) {
                            pu.points(egui_plot::Points::new(vec![[a[0], a[1]]]).radius(6.0_f32).name("start"));
                            pu.points(egui_plot::Points::new(vec![[b[0], b[1]]]).radius(6.0_f32).shape(egui_plot::MarkerShape::Square).name("finish"));
                        }
                    });
                }
                Tab::Elevation => {
                    let pv = self.preview.as_ref().unwrap();
                    Plot::new("elev").show(ui, |pu| {
                        let pts: Vec<[f64; 2]> = pv.pts.iter().zip(&pv.s).map(|(p, s)| [*s, p[2]]).collect();
                        pu.line(Line::new(PlotPoints::from(pts)).width(2.0_f32).name("elevation (m)"));
                        for j in &pv.joins {
                            pu.vline(VLine::new(*j).name("join"));
                        }
                    });
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 700.0]).with_title("ACR Remixer"),
        ..Default::default()
    };
    // the app id stays "acr-stagegen": eframe stores the saved settings under it
    eframe::run_native("acr-stagegen", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(test)]
mod tests {
    use super::forward_pipe_lines;

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
}
