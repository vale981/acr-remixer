// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-License-Identifier: GPL-3.0-or-later

//! Embed the frozen backend (packaging/dist/stagegen-backend.exe) when it has
//! been built; without it the GUI runs the Python backend (development).
//! Also records the git commit for the about/bug-report text.
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = root.join("../packaging/dist/stagegen-backend.exe");
    println!("cargo::rustc-check-cfg=cfg(embedded_backend)");
    println!("cargo:rerun-if-changed={}", p.display());
    if p.is_file() {
        println!("cargo:rustc-cfg=embedded_backend");
    }
    // commit (+ "-dirty" with uncommitted changes) and its date, e.g. "1ad28d0 2026-11-23"
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let hash =
        git(&["describe", "--always", "--dirty", "--abbrev=7"]).unwrap_or_else(|| "unknown".into());
    let date = git(&["log", "-1", "--format=%cs"]).unwrap_or_default();
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        format!("{hash} {date}").trim()
    );
    // HEAD only names the branch; the branch ref (or packed-refs) changes with every
    // commit, the index with staged changes (for -dirty)
    for f in ["HEAD", "packed-refs", "index"] {
        println!("cargo:rerun-if-changed=../.git/{f}");
    }
    if let Some(r) = std::fs::read_to_string(root.join("../.git/HEAD"))
        .ok()
        .and_then(|h| h.strip_prefix("ref: ").map(|s| s.trim().to_string()))
    {
        println!("cargo:rerun-if-changed=../.git/{r}");
    }
}
