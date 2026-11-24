// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-License-Identifier: GPL-3.0-or-later

//! `OodleLZ_Decompress`-compatible export backed by the open-source
//! oozextract decompressor (MIT), so acr-stagegen needs no proprietary Oodle
//! DLL. Same signature as oo2core's: returns the number of bytes written to
//! `dst`, or 0 on failure. Only the first four arguments are used.

use std::os::raw::{c_int, c_void};

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "system" fn OodleLZ_Decompress(
    src: *const c_void,
    src_len: i64,
    dst: *mut c_void,
    dst_len: i64,
    _fuzz: c_int,
    _crc: c_int,
    _verbose: c_int,
    _dst_base: *mut c_void,
    _dst_base_size: i64,
    _cb: *mut c_void,
    _cb_ctx: *mut c_void,
    _decoder_mem: *mut c_void,
    _decoder_mem_size: i64,
    _thread_phase: c_int,
) -> i64 {
    if src.is_null() || dst.is_null() || src_len <= 0 || dst_len <= 0 {
        return 0;
    }
    let input = std::slice::from_raw_parts(src as *const u8, src_len as usize);
    let output = std::slice::from_raw_parts_mut(dst as *mut u8, dst_len as usize);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oozextract::Extractor::new().read_from_slice(input, output)
    }));
    match res {
        Ok(Ok(n)) => n as i64,
        _ => 0,
    }
}
