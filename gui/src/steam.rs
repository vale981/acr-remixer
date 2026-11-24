// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-License-Identifier: GPL-3.0-or-later

//! Find the Assetto Corsa Rally install: Steam path from the registry, then
//! every library in steamapps/libraryfolders.vdf, then the app folder.

use std::path::{Path, PathBuf};

pub const APP_DIR: &str = "Assetto Corsa Rally";

fn steam_root() -> Option<PathBuf> {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    use winreg::RegKey;
    let tries = [
        (HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        (HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
    ];
    for (hive, key, val) in tries {
        if let Ok(k) = RegKey::predef(hive).open_subkey(key) {
            if let Ok(p) = k.get_value::<String, _>(val) {
                let p = PathBuf::from(p.replace('/', "\\"));
                if p.is_dir() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Library roots listed in libraryfolders.vdf ("path" entries).
fn libraries(steam: &Path) -> Vec<PathBuf> {
    let mut out = vec![steam.to_path_buf()];
    let vdf = steam.join("steamapps").join("libraryfolders.vdf");
    if let Ok(txt) = std::fs::read_to_string(vdf) {
        for line in txt.lines() {
            let parts: Vec<&str> = line.split('"').collect();
            // 	"path"		"E:\\SteamLibrary"
            if parts.len() >= 4 && parts[1] == "path" {
                let p = PathBuf::from(parts[3].replace("\\\\", "\\"));
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// A folder is the game if it contains the cooked paks.
pub fn is_acr_dir(p: &Path) -> bool {
    p.join("acr").join("Content").join("Paks").is_dir()
}

pub fn detect_acr() -> Option<PathBuf> {
    let steam = steam_root()?;
    libraries(&steam)
        .into_iter()
        .map(|lib| lib.join("steamapps").join("common").join(APP_DIR))
        .find(|p| is_acr_dir(p))
}
