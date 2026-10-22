#![forbid(unsafe_code)]

// This crate is a native-only CLI tool, but the workspace CI builds `--target wasm32-unknown-unknown
// --workspace --tests --no-run` to ensure the emulator crates remain wasm-compatible.
//
// Provide a tiny wasm32 stub `main` so the workspace continues to compile for wasm targets. The CLI
// is not expected to run in the browser.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fmt;
    use std::fs::File;
    use std::io::{self, BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use aero_machine::jit::IrBlockCache;
    use aero_machine::{BootDevice, Machine, MachineConfig, RunExit};
    use aero_storage::{AeroCowDisk, DiskImage, StdFileBackend, VirtualDisk, SECTOR_SIZE};
    use anyhow::{anyhow, bail, Context, Result};
    use clap::{ArgGroup, Parser, ValueEnum};

    const SLICE_INST_BUDGET: u64 = 100_000;

    #[derive(Debug, Parser)]
    #[command(
        about = "Native runner for aero_machine::Machine (boot/integration debugging)",
        group(
            ArgGroup::new("stop")
                .required(true)
                .args(["max_insts", "max_ms"])
        ),
        group(
            ArgGroup::new("media")
                .required(true)
                .multiple(true)
                .args(["disk", "install_iso"])
        )
    )]
    pub struct Args {
        /// Disk image to attach (raw/qcow2/vhd/aerospar; auto-detected).
        ///
        /// The virtual capacity must be a multiple of 512 bytes.
        #[arg(long)]
        disk: Option<PathBuf>,

        /// Open the disk image read-only (guest writes will fail).
        #[arg(long, conflicts_with = "disk_overlay", requires = "disk")]
        disk_ro: bool,

        /// Optional copy-on-write overlay image (AEROSPAR).
        ///
        /// If provided, the base `--disk` is opened read-only and guest writes go to the overlay.
        #[arg(long, requires = "disk")]
        disk_overlay: Option<PathBuf>,

        /// Allocation unit (block size) used when creating a new `--disk-overlay` (bytes).
        ///
        /// Must be a power of two and a multiple of 512.
        #[arg(long, default_value_t = 1024 * 1024, requires = "disk_overlay")]
        disk_overlay_block_size: u32,

        /// Guest RAM size in MiB.
        #[arg(long, default_value_t = 64)]
        ram: u64,

        /// Stop after executing at most N guest instructions.
        #[arg(long)]
        max_insts: Option<u64>,

        /// Stop after running for at most N milliseconds of host time.
        #[arg(long)]
        max_ms: Option<u64>,

        /// Where to write accumulated COM1 output bytes (`stdout` or a file path).
        #[arg(long, default_value = "stdout")]
        serial_out: String,

        /// Where to write accumulated DebugCon output bytes (I/O port `0xE9`).
        ///
        /// Use `none` to disable (default), `stdout` to write to stdout, or a file path.
        #[arg(long, default_value = "none")]
        debugcon_out: String,

        /// Dump the last VGA framebuffer to a PNG file on exit.
        #[arg(long)]
        vga_png: Option<PathBuf>,

        /// Re-dump the VGA framebuffer to `--vga-png` every N milliseconds of host time.
        ///
        /// Lets you watch a long boot progress (setup screens, desktop appearing) in
        /// near-real-time instead of only at the end. Requires `--vga-png`.
        #[arg(long, requires = "vga_png")]
        vga_png_interval: Option<u64>,

        /// Print RIP/mode/`total_executed` to stderr every N milliseconds of host time.
        ///
        /// Useful on multi-billion-instruction kernel probes where the VGA stays black
        /// between mode switches and you need proof of forward progress.
        #[arg(long)]
        progress_interval: Option<u64>,

        /// Enable IR-block JIT acceleration. Hot blocks are pre-decoded to IR
        /// and executed via the Tier-1 IR interpreter, skipping per-instruction
        /// decode and dispatch overhead.
        #[arg(long)]
        jit: bool,

        /// Save a snapshot (aero_snapshot format) on exit.
        #[arg(long)]
        snapshot_save: Option<PathBuf>,

        /// Load a snapshot (aero_snapshot format) before running.
        ///
        /// Works with either media type: the `media` arg group already enforces that one of
        /// `--disk` / `--install-iso` is present, so a CD-boot snapshot can be resumed with
        /// `--install-iso` (no hard `--disk` requirement).
        #[arg(long)]
        snapshot_load: Option<PathBuf>,

        /// Optional install/recovery ISO to attach as an ATAPI CD-ROM (IDE secondary master).
        ///
        /// This uses the canonical Win7 install-media slot (`disk_id=1`).
        #[arg(long)]
        install_iso: Option<PathBuf>,

        /// BIOS boot selection policy.
        ///
        /// Defaults to:
        /// - `hdd` when no `--install-iso` is provided
        /// - `cd-first` when both `--disk` and `--install-iso` are provided (install flow: boot CD once, then reboot into HDD)
        /// - `cdrom` when `--install-iso` is provided without `--disk` (ISO-only boots)
        ///
        /// Note: when `--snapshot-load` is used, this only affects future guest resets. The current
        /// CPU state is restored from the snapshot and the VM is not reset.
        #[arg(long, value_enum)]
        boot: Option<BootMode>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
    enum BootMode {
        /// Boot from the primary HDD (`DL=0x80`).
        Hdd,
        /// Boot from the install-media CD-ROM (`DL=0xE0`).
        Cdrom,
        /// Enable firmware "CD-first when present" policy (try `DL=0xE0` when ISO is present, otherwise fall back to HDD).
        CdFirst,
    }

    impl fmt::Display for BootMode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                BootMode::Hdd => f.write_str("hdd"),
                BootMode::Cdrom => f.write_str("cdrom"),
                BootMode::CdFirst => f.write_str("cd-first"),
            }
        }
    }

    fn browser_code_to_bios_key(code: &str) -> Option<u16> {
        // Classic INT 16h words: Set-1 scan code in AH, ASCII in AL.
        // Keep this intentionally small and aligned with AERO_INJECT_KEY's
        // documented bring-up keys.
        match code {
            "Enter" => Some(0x1c0d),
            "Space" => Some(0x3920),
            "Escape" => Some(0x011b),
            "Tab" => Some(0x0f09),
            _ => None,
        }
    }

    /// Periodic re-inject is for a live Setup dialog. An idle HLT snapshot
    /// would otherwise see a new press every ~50 zero-inst slices.
    fn inject_key_repeats_each_slice(once: bool) -> bool {
        !once
    }

    /// `AERO_DUMP_THREADS=*`, a single name, or a comma list.
    fn thread_filter_matches(filter: &str, image_name: &str) -> bool {
        filter == "*"
            || filter
                .split(',')
                .any(|part| part.eq_ignore_ascii_case(image_name))
    }

    /// Win7 SP1 x64 `EPROCESS.ImageFileName` is a 15-byte padded ASCII tag.
    /// `DUMP_PROCS` is a 2 GiB physical scan, so the gate must stay cheap,
    /// but it must not be a case-sensitive allow-list: first-boot specialize
    /// is `Setup.exe` (capital S), and helpers (`windeploy`, `drvinst`,
    /// `WerFault`, truncated `SearchIndexer.`) were previously invisible.
    fn is_known_eprocess_image_name(image: &[u8]) -> bool {
        is_plausible_eprocess_image_name(image)
    }

    fn is_plausible_eprocess_image_name(image: &[u8]) -> bool {
        if image.len() != 15 {
            return false;
        }
        if !image[0].is_ascii_alphabetic() {
            return false;
        }
        if !image.iter().all(|&b| b == 0 || b.is_ascii_graphic()) {
            return false;
        }
        let end = image.iter().position(|&b| b == 0).unwrap_or(15);
        if end == 0 || image[end..].iter().any(|&b| b != 0) {
            return false;
        }
        let name = &image[..end];
        if name.eq_ignore_ascii_case(b"Idle") || name.eq_ignore_ascii_case(b"System") {
            return true;
        }
        // File-cache / CBS / path strings are full of dots; an EPROCESS tag is
        // a single Win32 image name (`Setup.exe`) or a 15-byte truncation
        // (`SearchIndexer.exe` → `SearchIndexer.`).
        if name.iter().any(|&b| {
            matches!(
                b,
                b'\\'
                    | b'/'
                    | b'|'
                    | b'~'
                    | b' '
                    | b'#'
                    | b'{'
                    | b'}'
                    | b'+'
                    | b'('
                    | b')'
                    | b'['
                    | b']'
                    | b','
                    | b':'
                    | b'*'
                    | b'?'
            )
        }) {
            return false;
        }
        let Some(dot) = name.iter().rposition(|&b| b == b'.') else {
            return false;
        };
        let stem = &name[..dot];
        let ext = &name[dot + 1..];
        if stem.len() < 2
            || !stem
                .iter()
                .all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return false;
        }
        let ext_l = ext.to_ascii_lowercase();
        ext_l == b"exe"
            || ext_l == b"com"
            || ext_l == b"scr"
            // 15-byte field truncates `SearchIndexer.exe` to `SearchIndexer.`
            // (NUL-padded) or `SearchIndexer.e` (no room for a NUL).
            || (stem.len() >= 8
                && (ext_l.is_empty()
                    || b"exe".starts_with(ext_l.as_slice())
                    || b"com".starts_with(ext_l.as_slice())
                    || b"scr".starts_with(ext_l.as_slice())))
    }

    #[cfg(test)]
    mod inject_key_once_tests {
        use super::{browser_code_to_bios_key, inject_key_repeats_each_slice};

        #[test]
        fn once_disables_periodic_reinject_on_idle_hlt() {
            assert!(
                !inject_key_repeats_each_slice(true),
                "AERO_INJECT_KEY_ONCE must not re-fire on idle slices"
            );
            assert!(inject_key_repeats_each_slice(false));
        }

        #[test]
        fn escape_is_a_one_shot_bios_key() {
            assert_eq!(browser_code_to_bios_key("Escape"), Some(0x011b));
        }
    }

    #[cfg(test)]
    mod eprocess_name_tests {
        use super::is_known_eprocess_image_name;

        fn pad15(name: &str) -> [u8; 15] {
            let mut out = [0u8; 15];
            out[..name.len()].copy_from_slice(name.as_bytes());
            out
        }

        #[test]
        fn dump_procs_sees_vds_loader_and_service() {
            assert!(
                is_known_eprocess_image_name(&pad15("vdsldr.exe")),
                "COM local-server launch is invisible if DUMP_PROCS drops vdsldr.exe"
            );
            assert!(is_known_eprocess_image_name(&pad15("vds.exe")));
            assert!(is_known_eprocess_image_name(&pad15("lsm.exe")));
            assert!(is_known_eprocess_image_name(&pad15("setup.exe")));
            assert!(
                is_known_eprocess_image_name(&pad15("setupcl.exe")),
                "first-boot specialize is invisible if DUMP_PROCS drops setupcl.exe"
            );
            assert!(
                is_known_eprocess_image_name(&pad15("Setup.exe")),
                "HDD first-boot paints Setup.exe (capital S); a lowercase-only gate hides it"
            );
            assert!(is_known_eprocess_image_name(&pad15("windeploy.exe")));
            assert!(is_known_eprocess_image_name(&pad15("drvinst.exe")));
            assert!(is_known_eprocess_image_name(&pad15("WerFault.exe")));
            assert!(
                is_known_eprocess_image_name(&pad15("SearchIndexer.")),
                "ImageFileName is 15 bytes; SearchIndexer.exe truncates to SearchIndexer."
            );
            assert!(
                !is_known_eprocess_image_name(&pad15("notareal")),
                "graphic junk without a dot is not an EPROCESS image tag"
            );
            assert!(!is_known_eprocess_image_name(&[0u8; 15]));
            assert!(
                !is_known_eprocess_image_name(&pad15("32\\mswsock.dll")),
                "path fragments in the file cache must not match"
            );
            assert!(!is_known_eprocess_image_name(&pad15("4e35~amd64~~6.1")));
            assert!(!is_known_eprocess_image_name(&pad15("dme.txt")));
        }
    }

    #[cfg(test)]
    mod dump_threads_filter_tests {
        use super::thread_filter_matches;

        #[test]
        fn star_and_comma_list_select_setup_and_vds() {
            assert!(thread_filter_matches("*", "vdsldr.exe"));
            assert!(thread_filter_matches("setup.exe,vds.exe", "vds.exe"));
            assert!(!thread_filter_matches("setup.exe,vds.exe", "wmiprvse.exe"));
        }
    }

    /// Build a machine whose guest RAM lives in a `WasmtimeBackend`'s linear memory.
    ///
    /// Returns the backend alongside the machine because it owns that memory:
    /// dropping it would invalidate the pointer the machine's RAM is built on.
    fn build_machine_on_jit_ram(
        cfg: MachineConfig,
    ) -> Result<(
        Machine,
        std::rc::Rc<
            std::cell::RefCell<
                aero_jit_x86::backend::WasmtimeBackend<aero_cpu_core::state::CpuState>,
            >,
        >,
    )> {
        use aero_machine::jit_ram::{WasmGuestRam, RAM_BASE};

        let ram_bytes = cfg.ram_size_bytes;
        let total = RAM_BASE
            .checked_add(ram_bytes)
            .ok_or_else(|| anyhow!("guest RAM too large for a wasm32 linear memory"))?;
        let pages = total.div_ceil(65_536);
        let pages = u32::try_from(pages)
            .map_err(|_| anyhow!("guest RAM needs {pages} wasm pages, more than wasm32 allows"))?;

        let backend = std::rc::Rc::new(std::cell::RefCell::new(
            aero_jit_x86::backend::WasmtimeBackend::new_with_memory_pages(pages, 0x100),
        ));
        let ram = WasmGuestRam::new(backend.clone(), ram_bytes).ok_or_else(|| {
            anyhow!("backend linear memory too small for {ram_bytes} bytes of RAM")
        })?;
        eprintln!(
            "AERO_JIT_RAM: guest RAM is wasm linear memory ({pages} pages, ram_base={RAM_BASE:#x})"
        );

        let machine =
            Machine::new_with_guest_memory(cfg, Box::new(ram)).map_err(|e| anyhow!("{e}"))?;
        Ok((machine, backend))
    }

    pub fn main() -> Result<()> {
        let args = Args::parse();

        let ram_bytes = args
            .ram
            .checked_mul(1024 * 1024)
            .context("RAM size overflow")?;

        // By default:
        // - If install media + HDD are both present, boot CD once (CD-first policy) and then allow
        //   the guest to reboot into HDD without host-side boot-drive toggling (mirrors the browser
        //   runtime's `vmRuntime="machine"` install flow).
        // - If only install media is present, boot from CD directly (no CD-first toggling).
        let boot_mode = args.boot.unwrap_or_else(|| {
            if args.install_iso.is_some() && args.disk.is_some() {
                BootMode::CdFirst
            } else if args.install_iso.is_some() {
                BootMode::Cdrom
            } else {
                BootMode::Hdd
            }
        });
        if matches!(boot_mode, BootMode::Cdrom | BootMode::CdFirst) && args.install_iso.is_none() {
            bail!("--boot={boot_mode} requires --install-iso");
        }
        if matches!(boot_mode, BootMode::Hdd | BootMode::CdFirst) && args.disk.is_none() {
            bail!("--boot={boot_mode} requires --disk");
        }

        // Win7 bring-up needs AeroGPU (BAR1 VBE LFB + legacy aliasing), not the
        // transitional standalone VGA-only path. `win7_storage_defaults` is
        // storage-topology only and leaves `enable_vga=true` / `enable_aerogpu=false`,
        // which never paints bootvid/setup graphics in the host present path.
        //
        // `AERO_LEGACY_VGA=1` forces the standalone VGA path for diagnosis when
        // resuming snapshots that were captured under `enable_vga=true`.
        // Set when guest RAM is in the JIT's linear memory *and* compiled execution
        // is enabled; the tier shares the backend that owns that memory.
        let mut compiled_tier: Option<aero_machine::compiled_tier::NativeCompiledTier> = None;
        let mut machine = if std::env::var_os("AERO_LEGACY_VGA").is_some() {
            eprintln!("AERO_LEGACY_VGA: using MachineConfig::win7_storage (standalone VGA)");
            Machine::new(MachineConfig::win7_storage(ram_bytes)).map_err(|e| anyhow!("{e}"))?
        } else {
            if std::env::var_os("AERO_AEROGPU_STDVGA_IDS").is_some() {
                eprintln!(
                    "AERO_AEROGPU_STDVGA_IDS: AeroGPU 00:07.0 reports Bochs Standard VGA 1234:1111 identity/topology"
                );
            }
            let cfg = MachineConfig::win7_graphics(ram_bytes);
            if std::env::var_os("AERO_JIT_RAM").is_some() {
                // Put guest RAM inside the JIT's WASM linear memory. Compiled Tier-1
                // blocks address RAM as linear-memory offsets, so this is what lets a
                // guest load become a load *inside* the compiled block rather than a
                // call back out to the host. Nothing executes compiled code yet: this
                // establishes that the machine runs unchanged on that backing.
                let (machine, backend) =
                    build_machine_on_jit_ram(cfg).map_err(|e| anyhow!("{e}"))?;
                // The backend owns the linear memory the machine's RAM points into,
                // so it has to outlive the machine. Hand it to the compiled tier
                // when that is enabled, and otherwise keep it alive here.
                if std::env::var_os("AERO_COMPILED_TIER").is_some() {
                    compiled_tier = Some(aero_machine::compiled_tier::CompiledTier::new(backend));
                } else {
                    std::mem::forget(backend);
                }
                machine
            } else {
                Machine::new(cfg).map_err(|e| anyhow!("{e}"))?
            }
        };

        // Record the host's chosen disk paths in the machine's snapshot overlay refs so snapshots
        // produced by this CLI remain self-describing (even when no explicit COW overlay is used).
        //
        // Note: these refs are metadata only; disk bytes always remain external to the snapshot
        // blob.
        let base_image = args
            .disk
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        if let Some(disk) = &args.disk {
            if let Some(overlay) = &args.disk_overlay {
                machine.set_ahci_port0_disk_overlay_ref(
                    base_image.clone(),
                    overlay.display().to_string(),
                );
            } else {
                machine.set_ahci_port0_disk_overlay_ref(base_image.clone(), "");
            }

            let disk_backend = if let Some(overlay) = &args.disk_overlay {
                open_disk_backend_with_overlay(disk, overlay, args.disk_overlay_block_size)?
            } else {
                open_disk_backend(disk, args.disk_ro)?
            };
            machine
                .set_disk_backend(disk_backend)
                .map_err(|e| anyhow!("{e}"))?;
        }

        // Track whether the firmware "CD-first when present" policy is enabled so we can disable it
        // after the first guest-initiated reset (Windows setup reboots into the installed HDD while
        // leaving install media inserted).
        let mut cd_first_enabled: bool;

        if let Some(path) = &args.snapshot_load {
            let mut f = File::open(path)
                .with_context(|| format!("failed to open snapshot for load: {}", path.display()))?;
            machine
                .restore_snapshot_from_checked(&mut f)
                .map_err(|e| anyhow!("{e}"))?;

            let tty = machine.bios_tty_output_bytes();
            if !tty.is_empty() {
                eprintln!(
                    "[TTY-drain] BIOS TTY buffer ({} bytes):\n{}",
                    tty.len(),
                    String::from_utf8_lossy(&tty)
                );
            } else {
                eprintln!("[TTY-drain] BIOS TTY buffer is empty");
            }

            // Snapshot disk refs are host-managed metadata. Warn if the snapshot was produced for a
            // different base/overlay path than the current CLI flags.
            if let Some(restored) = machine.restored_disk_overlays() {
                if let Some(primary) = restored
                    .disks
                    .iter()
                    .find(|d| d.disk_id == Machine::DISK_ID_PRIMARY_HDD)
                {
                    if !primary.base_image.is_empty() && primary.base_image != base_image {
                        eprintln!(
                            "warning: snapshot base_image differs from --disk: snapshot={} cli={}",
                            primary.base_image, base_image
                        );
                    }
                    if !primary.overlay_image.is_empty() {
                        match &args.disk_overlay {
                            Some(cli_overlay) => {
                                let cli_overlay = cli_overlay.display().to_string();
                                if primary.overlay_image != cli_overlay {
                                    eprintln!(
                                        "warning: snapshot overlay_image differs from --disk-overlay: snapshot={} cli={}",
                                        primary.overlay_image, cli_overlay
                                    );
                                }
                            }
                            None => {
                                eprintln!(
                                    "warning: snapshot specifies overlay_image {} but CLI did not provide --disk-overlay",
                                    primary.overlay_image
                                );
                            }
                        }
                    }
                }

                if let Some(install_iso) = &args.install_iso {
                    let cli_iso = install_iso.display().to_string();
                    if let Some(cd) = restored
                        .disks
                        .iter()
                        .find(|d| d.disk_id == Machine::DISK_ID_INSTALL_MEDIA)
                    {
                        if !cd.base_image.is_empty() && cd.base_image != cli_iso {
                            eprintln!(
                                "warning: snapshot install-media base_image differs from --install-iso: snapshot={} cli={}",
                                cd.base_image, cli_iso
                            );
                        }
                        if !cd.overlay_image.is_empty() {
                            eprintln!(
                                "warning: snapshot install-media overlay_image is non-empty (expected read-only ISO): {}",
                                cd.overlay_image
                            );
                        }
                    }
                }
            }

            // Storage controller snapshots intentionally drop host backends. Reattach the shared
            // disk so the guest can continue booting after restore.
            machine
                .attach_shared_disk_to_ahci_port0()
                .context("failed to reattach shared disk to AHCI port0")?;
            machine
                .attach_shared_disk_to_virtio_blk()
                .map_err(|e| anyhow!("{e}"))?;

            // If install media is provided, reattach it without changing guest-visible tray state.
            if let Some(iso_path) = &args.install_iso {
                let iso = open_disk_image(iso_path, true)?;
                machine
                    .attach_install_media_iso_for_restore(Box::new(iso))
                    .with_context(|| {
                        format!(
                            "failed to attach install ISO for restore: {}",
                            iso_path.display()
                        )
                    })?;
                machine
                    .set_ide_secondary_master_atapi_overlay_ref(iso_path.display().to_string(), "");
            }

            // Only override the snapshot's boot policy when explicitly requested. Otherwise we keep
            // the restored BIOS config intact.
            if args.boot.is_some() {
                match boot_mode {
                    BootMode::Hdd => {
                        machine.set_boot_from_cd_if_present(false);
                        machine.set_boot_drive(0x80);
                    }
                    BootMode::Cdrom => {
                        machine.set_boot_from_cd_if_present(false);
                        machine.set_boot_drive(0xE0);
                    }
                    BootMode::CdFirst => {
                        machine.set_cd_boot_drive(0xE0);
                        machine.set_boot_from_cd_if_present(true);
                        machine.set_boot_drive(0x80);
                    }
                }
            }
            cd_first_enabled = machine.boot_from_cd_if_present();
        } else {
            // No snapshot restore: attach optional install media and apply boot policy, then reset.
            if let Some(iso_path) = &args.install_iso {
                let iso = open_disk_image(iso_path, true)?;
                machine
                    .attach_install_media_iso_and_set_overlay_ref(
                        Box::new(iso),
                        iso_path.display().to_string(),
                    )
                    .with_context(|| {
                        format!("failed to attach install ISO: {}", iso_path.display())
                    })?;
            }

            match boot_mode {
                BootMode::Hdd => {
                    machine.set_boot_from_cd_if_present(false);
                    machine.set_boot_drive(0x80);
                }
                BootMode::Cdrom => {
                    machine.set_boot_from_cd_if_present(false);
                    machine.set_boot_drive(0xE0);
                }
                BootMode::CdFirst => {
                    machine.set_cd_boot_drive(0xE0);
                    machine.set_boot_from_cd_if_present(true);
                    machine.set_boot_drive(0x80);
                }
            }
            cd_first_enabled = machine.boot_from_cd_if_present();

            // `Machine::new` performs an initial BIOS POST + boot attempt. Re-run POST after
            // attaching disks and configuring boot policy so the guest starts executing from the
            // selected boot device.
            machine.reset();
        }

        if std::env::var_os("AERO_DUMP_BOOT").is_some() {
            eprintln!(
                "AERO_DUMP_BOOT: configured_drive={:#04x} cd_drive={:#04x} cd_first={} \
                 active={:?} install_media_inserted={}",
                machine.boot_drive(),
                machine.cd_boot_drive(),
                machine.boot_from_cd_if_present(),
                machine.active_boot_device(),
                machine.install_media_is_inserted(),
            );
        }

        // Narrow standard-VGA PCI diagnostics for restored checkpoints. Keep each override
        // independently selectable so QEMU topology differences can be isolated one at a time.
        let stdvga_enable_io = std::env::var_os("AERO_STDVGA_ENABLE_IO").is_some();
        let stdvga_revision = std::env::var_os("AERO_STDVGA_REVISION");
        if stdvga_enable_io || stdvga_revision.is_some() {
            let bdf = machine
                .aerogpu_bdf()
                .context("AERO_STDVGA_* diagnostics require AeroGPU/stdvga at its canonical BDF")?;
            let pci_cfg = machine
                .pci_config_ports()
                .context("AERO_STDVGA_* diagnostics require PCI config ports")?;
            let mut pci_cfg = pci_cfg.borrow_mut();
            let bus = pci_cfg.bus_mut();
            let id = bus.read_config(bdf, 0x00, 4);
            if id != 0x1111_1234 {
                bail!(
                    "AERO_STDVGA_* diagnostics require standard-VGA ID 1234:1111, got {:04x}:{:04x}",
                    id as u16,
                    id >> 16
                );
            }
            let class = bus.read_config(bdf, 0x08, 4);
            let base_class = (class >> 24) as u8;
            let subclass = (class >> 16) as u8;
            if (base_class, subclass) != (0x03, 0x00) {
                bail!(
                    "AERO_STDVGA_* diagnostics require VGA-compatible class 03/00, got {base_class:02x}/{subclass:02x}"
                );
            }
            if stdvga_enable_io {
                let old = bus.read_config(bdf, 0x04, 2) as u16;
                bus.write_config(bdf, 0x04, 2, u32::from(old | 0x0001));
                let new = bus.read_config(bdf, 0x04, 2) as u16;
                eprintln!(
                    "AERO_STDVGA_ENABLE_IO: diagnostic PCI command override at {bdf:?}: {old:#06x} -> {new:#06x}"
                );
            }
            if let Some(raw) = stdvga_revision {
                let raw = raw
                    .into_string()
                    .map_err(|_| anyhow!("AERO_STDVGA_REVISION must be valid UTF-8"))?;
                let revision =
                    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
                        u8::from_str_radix(hex, 16)
                    } else {
                        raw.parse::<u8>()
                    }
                    .with_context(|| {
                        format!("AERO_STDVGA_REVISION must be an 8-bit integer, got {raw:?}")
                    })?;
                let cfg = bus
                    .device_config_mut(bdf)
                    .context("standard-VGA PCI function disappeared during revision override")?;
                let class = cfg.class_code();
                cfg.set_class_code(class.class, class.subclass, class.prog_if, revision);
                eprintln!(
                    "AERO_STDVGA_REVISION: diagnostic PCI revision override at {bdf:?}: {:#04x} -> {revision:#04x}",
                    class.revision_id
                );
            }
        }

        let mut serial_sink = open_serial_sink(&args.serial_out)?;
        let mut debugcon_sink = open_optional_sink(&args.debugcon_out)?;

        // Bring-up poke: AERO_POKE_LINEAR=<va>:<hexbytes>[,...] writes bytes through the
        // guest page tables. Re-applied every host slice (sticky): a one-shot write under
        // winload CR3 is lost when the kernel remaps the VA (HAL clock-wait flag for
        // 0x5C/0x10B). Soft-fail while unmapped; log first success and CR3 changes.
        let poke_spec = std::env::var("AERO_POKE_LINEAR")
            .ok()
            .filter(|s| !s.is_empty());
        let mut poke_logged_ok = false;
        let mut poke_last_cr3: Option<u64> = None;
        if let Some(spec) = poke_spec.as_deref() {
            match poke_linear_regions(&mut machine, spec) {
                Ok(msg) => {
                    eprintln!("AERO_POKE_LINEAR:{msg}");
                    poke_logged_ok = true;
                    poke_last_cr3 = Some(machine.cpu().control.cr3);
                }
                Err(e) => {
                    eprintln!("AERO_POKE_LINEAR: deferred ({e}); sticky retry each slice");
                }
            }
        }
        // Physical poke (no paging): AERO_POKE_PHYS=<pa>:<hexbytes>[,...]
        // Large payloads: AERO_POKE_PHYS_FILE=<path> (same grammar; avoids ARG_MAX/env limits).
        let poke_phys_spec = std::env::var("AERO_POKE_PHYS")
            .ok()
            .filter(|s| !s.is_empty());
        let poke_phys_file = std::env::var("AERO_POKE_PHYS_FILE")
            .ok()
            .filter(|s| !s.is_empty());
        let poke_phys_from_file =
            poke_phys_file
                .as_ref()
                .and_then(|path| match std::fs::read_to_string(path) {
                    Ok(s) => {
                        let s = s.trim().to_string();
                        if s.is_empty() {
                            None
                        } else {
                            Some(s)
                        }
                    }
                    Err(e) => {
                        eprintln!("AERO_POKE_PHYS_FILE: failed to read {path}: {e}");
                        None
                    }
                });
        let poke_phys_combined = match (poke_phys_spec.as_deref(), poke_phys_from_file.as_deref()) {
            (Some(a), Some(b)) => Some(format!("{a},{b}")),
            (Some(a), None) => Some(a.to_string()),
            (None, Some(b)) => Some(b.to_string()),
            (None, None) => None,
        };
        if let Some(spec) = poke_phys_combined.as_deref() {
            match poke_phys_regions(&mut machine, spec) {
                Ok(msg) => eprintln!("AERO_POKE_PHYS:{msg}"),
                Err(e) => eprintln!("AERO_POKE_PHYS: {e:#}"),
            }
        }

        // Durable type-5 (HOOK) handle-table wedge: scan the win32k session handle
        // table and free every type-5 entry whose object is null or whose HEAD.h is 0.
        // Single-byte poke at entry 198 was enough for the first winpeshl wall; cold
        // stdvga/winlogon hits more corrupt HOOK entries → multi-entry scan.
        // Env: AERO_AUTO_TYPE5_WEDGE=1 (optional AERO_TYPE5_TABLE_BASE / AERO_TYPE5_MAX).
        let auto_type5 = std::env::var_os("AERO_AUTO_TYPE5_WEDGE").is_some();
        if auto_type5 {
            match wedge_type5_handle_table(&mut machine) {
                Ok(msg) => eprintln!("AERO_AUTO_TYPE5_WEDGE:{msg}"),
                Err(e) => {
                    eprintln!("AERO_AUTO_TYPE5_WEDGE: deferred ({e}); sticky retry each slice")
                }
            }
        }

        // WinPE winpeshl SCE wall: WpeInstallServicesSecurityTemplate (and sibling
        // WpeUtil imports) block forever on \\.\PIPE\scerpc when scesrv never
        // answers. Redirect all WpeUtil IAT slots to `xor eax,eax; ret` so winpeshl
        // skips SCE and proceeds to CreateProcess(setup). Sticky each slice once
        // winpeshl.exe is present (ImageBase from PEB). AERO_AUTO_SCE_WEDGE=1.
        let auto_sce = std::env::var_os("AERO_AUTO_SCE_WEDGE").is_some();
        if auto_sce {
            match wedge_sce_wpeutil_iat(&mut machine) {
                Ok(msg) => eprintln!("AERO_AUTO_SCE_WEDGE:{msg}"),
                Err(e) => eprintln!("AERO_AUTO_SCE_WEDGE: deferred ({e}); sticky retry each slice"),
            }
        }

        // Paint-wall wedge: identity-map the VBE LFB GPA into the *current* CR3 as a
        // 2 MiB large page (PCD/PWT). Proved necessary because post-boot guest page
        // tables leave 0xe4040000 NP everywhere (System/winpeshl/csrss), so GDI cannot
        // touch VRAM (`AERO_COUNT_LFB_WRITES` stays 0). `AERO_MAP_LFB=1` enables.
        // Sticky each slice: CR3 switches (setup/csrss/winlogon) drop a one-shot map.
        let auto_map_lfb = std::env::var_os("AERO_MAP_LFB").is_some();
        if auto_map_lfb {
            match map_vbe_lfb_identity_2m(&mut machine) {
                Ok(msg) => eprintln!("AERO_MAP_LFB:{msg}"),
                Err(e) => eprintln!("AERO_MAP_LFB: deferred ({e}); sticky retry each slice"),
            }
        }
        // Remap Eng FrameBufferBase VA(s) onto VBE LFB phys. Miniport may keep
        // phys=e4040000 + a system-PTE VA while the PDE is NP (session thrash).
        // `AERO_MAP_LFB_VA=<hex>[,hex…]` sticky each slice.
        //
        // WARNING: forging MMIO PFNs into ordinary page tables causes Win7
        // MEMORY_MANAGEMENT 0x1A/0x5100 on long runs (Mm PFN/bitmap validation).
        // Use only for short path-proof fills; prefer `AERO_REDIR_ENG_LFB` which
        // points FrameBufferBase/SURFOBJ at the live bootvid KVA (Windows-owned).
        let map_lfb_vas: Vec<u64> = std::env::var("AERO_MAP_LFB_VA")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.split(',')
                    .filter_map(|p| {
                        let p = p.trim().trim_start_matches("0x").trim_start_matches("0X");
                        u64::from_str_radix(p, 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !map_lfb_vas.is_empty() {
            eprintln!(
                "AERO_MAP_LFB_VA: warning — sticky MMIO PTEs can bugcheck 0x1A/0x5100; short fills only"
            );
        }
        for &va in &map_lfb_vas {
            match map_va_to_vbe_lfb_2m(&mut machine, va) {
                Ok(msg) => eprintln!("AERO_MAP_LFB_VA:{msg}"),
                Err(e) => eprintln!("AERO_MAP_LFB_VA: deferred va={va:#x} ({e}); sticky retry"),
            }
        }
        // Safe present-path wedge: rewrite Eng FrameBufferBase + primary SURFOBJ
        // bits to the live bootvid MmMapIoSpace VA (no forged MMIO PTEs → no 0x1A).
        // `AERO_REDIR_ENG_LFB=1`. One full scan, then sticky re-poke cached phys addrs.
        let auto_redir_eng = std::env::var_os("AERO_REDIR_ENG_LFB").is_some();
        let mut redir_eng_targets: Option<(Option<u64>, Option<u64>)> = None;
        if auto_redir_eng {
            match redir_eng_lfb_to_bootvid_kva(&mut machine, None) {
                Ok((msg, eng_pa, surf_pa)) => {
                    eprintln!("AERO_REDIR_ENG_LFB:{msg}");
                    redir_eng_targets = Some((eng_pa, surf_pa));
                }
                Err(e) => eprintln!("AERO_REDIR_ENG_LFB: deferred ({e}); sticky retry"),
            }
        }

        // Reverse page-table search: find guest VAs that map a given physical address
        // (e.g. the VBE LFB). MmMapIoSpace creates high-half kernel VAs, not identity
        // maps — so `DUMP_LINEAR` of the GPA is the wrong probe. `AERO_FIND_PHYS=<pa>`.
        if let Some(spec) = std::env::var("AERO_FIND_PHYS")
            .ok()
            .filter(|s| !s.is_empty())
        {
            match find_phys_maps(&mut machine, &spec) {
                Ok(msg) => eprintln!("AERO_FIND_PHYS:{msg}"),
                Err(e) => eprintln!("AERO_FIND_PHYS: {e:#}"),
            }
        }

        // Bring-up: force a one-shot user-mode resume (abort in-flight kernel waits).
        // AERO_FORCE_RESUME=cr3=<hex>,rip=<hex>,rsp=<hex>[,rax=<hex>][,rcx=<hex>][,rdx=<hex>]
        // Switches the BSP into long-mode CPL3 under the given CR3 with Win7-typical
        // user selectors (CS=0x33, SS/DS/ES=0x2B), RFLAGS.IF=1, CR8=0, halted=false.
        // GS_BASE is set to the process TEB if AERO_FORCE_TEB is provided, else left.
        if let Some(spec) = std::env::var("AERO_FORCE_RESUME")
            .ok()
            .filter(|s| !s.is_empty())
        {
            match force_user_resume(&mut machine, &spec) {
                Ok(msg) => eprintln!("AERO_FORCE_RESUME:{msg}"),
                Err(e) => eprintln!("AERO_FORCE_RESUME: {e:#}"),
            }
        }

        // Wall-clock TSC: KeStallExecutionProcessor-style RDTSC busy-waits otherwise
        // require billions of host instructions (1 cycle/inst at 3 GHz). With wall-clock
        // scaling, a few-second guest stall completes in real seconds.
        // Default OFF: during early boot wall-clock TSC races ahead of retired
        // instructions and floods the guest with timer IRQs (measured ~2.6 vs ~6
        // Minst/s). Enable with AERO_TSC_WALLCLOCK=1 for post-kernel stall probes,
        // or jump past a known deadline with AERO_SET_TSC.
        let wallclock = matches!(
            std::env::var("AERO_TSC_WALLCLOCK"),
            Ok(v) if matches!(v.as_str(), "1" | "true" | "on" | "yes")
        );
        if wallclock {
            let core = machine.cpu_core_mut_by_index(0);
            let hz = core.time.tsc_hz();
            let tsc = core.time.read_tsc();
            core.time = aero_cpu_core::time::TimeSource::new_wallclock(hz);
            core.time.set_tsc(tsc);
            core.state.msr.tsc = tsc;
            eprintln!("AERO_TSC_WALLCLOCK: enabled (hz={hz}, tsc={tsc:#x})");
        }

        // Inject ACPI PM1 power-button status (bring-up: HAL polls PM1a for bit 8).
        // Use PWRBTN only — Win7 HAL requires bit 0x100 set AND bit 0x8000 (WAK) clear.
        // `acpi_wake()` sets both and permanently fails that check.
        if matches!(
            std::env::var("AERO_ACPI_PWRBTN").as_deref(),
            Ok("1") | Ok("true") | Ok("on") | Ok("yes")
        ) {
            if machine.acpi_pm().is_some() {
                machine.acpi_power_button();
                eprintln!("AERO_ACPI_PWRBTN: latched PM1_STS.PWRBTN only (no WAK)");
            } else {
                eprintln!("AERO_ACPI_PWRBTN: no ACPI PM device");
            }
        }

        // Bring-up: force architectural TSC (e.g. jump past a KeStall deadline).
        // AERO_SET_TSC=<hex|dec>
        if let Some(spec) = std::env::var("AERO_SET_TSC").ok().filter(|s| !s.is_empty()) {
            let tsc = if let Some(hex) = spec.strip_prefix("0x").or_else(|| spec.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16)
                    .with_context(|| format!("bad AERO_SET_TSC hex: {spec}"))?
            } else {
                spec.parse::<u64>()
                    .with_context(|| format!("bad AERO_SET_TSC: {spec}"))?
            };
            let core = machine.cpu_core_mut_by_index(0);
            core.time.set_tsc(tsc);
            core.state.msr.tsc = tsc;
            eprintln!("AERO_SET_TSC: {tsc:#x}");
        }

        // Tight-loop RDTSC scheduler yielding is env-gated in aero-cpu-core
        // (`AERO_RDTSC_QUANTUM`, default 0). The retained numeric spelling is compatibility with
        // older bring-up recipes; RDTSC remains a pure architectural sample and the value now
        // enables loop recognition/yielding rather than adding TSC cycles.
        if std::env::var_os("AERO_RDTSC_QUANTUM").is_none() {
            eprintln!(
                "AERO_RDTSC_QUANTUM: unset (default 0); set 2000 to yield sustained interruptible RDTSC polling loops"
            );
        } else if let Ok(value) = std::env::var("AERO_RDTSC_QUANTUM") {
            eprintln!(
                "AERO_RDTSC_QUANTUM: {value} enables tight-loop scheduler yields (architectural TSC is not fast-forwarded)"
            );
        }
        if let Ok(quantum) = std::env::var("AERO_PM_TIMER_POLL_QUANTUM") {
            eprintln!(
                "AERO_PM_TIMER_POLL_QUANTUM: {quantum} TSC cycles per PM_TMR poll (coherent platform-time fast-forward)"
            );
        }

        // Bring-up: force RFLAGS.IF=1 so HAL clock-sync waits can observe RTC/PIC IRQs.
        // Win7 HAL_INITIALIZATION_FAILED (0x5C / 0x10B) is a RDTSC-bounded wait for a
        // software flag incremented by the clock ISR; with IF=0 that flag never moves.
        // Re-applied each slice below — guest CLI inside HalpInit clears IF again.
        let force_if = matches!(
            std::env::var("AERO_FORCE_IF").as_deref(),
            Ok("1") | Ok("true") | Ok("on") | Ok("yes")
        );
        if force_if {
            let core = machine.cpu_core_mut_by_index(0);
            core.state.set_flag(aero_cpu_core::state::RFLAGS_IF, true);
            eprintln!("AERO_FORCE_IF: RFLAGS.IF sticky (re-set each slice)");
        }

        // Bring-up: force CR8 (architectural IRQL) each host slice.
        // Early HAL 0x5C / post-bugcheck HIGH_LEVEL hangs sometimes need CR8=0 so the
        // scheduler can run again. WARNING: sticky CR8=0 after session/multi-thread is
        // up is poison — interrupt poll mirrors CR8→LAPIC TPR, so forced CR8=0 unmasks
        // IRQs while the guest believes it raised IRQL for a DPC/critical section, and
        // Win7 then bugchecks ATTEMPTED_SWITCH_FROM_DPC (0xB8) (services.exe→System).
        // Prefer unset once past HAL init; only use for targeted early-boot wedges.
        // AERO_FORCE_CR8=<hex> (default unset = no override). Re-applied each slice.
        let force_cr8: Option<u64> = std::env::var("AERO_FORCE_CR8")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| {
                let s = s.trim().trim_start_matches("0x").trim_start_matches("0X");
                u64::from_str_radix(s, 16).ok()
            });
        if let Some(v) = force_cr8 {
            let core = machine.cpu_core_mut_by_index(0);
            core.state.control.cr8 = v;
            if v == 0 {
                eprintln!(
                    "AERO_FORCE_CR8: sticky cr8=0x0 (re-set each slice); \
                     WARN post-session use → 0xB8 ATTEMPTED_SWITCH_FROM_DPC"
                );
            } else {
                eprintln!("AERO_FORCE_CR8: sticky cr8={v:#x} (re-set each slice)");
            }
        }

        // Bring-up: inject browser-code keys into both PS/2 and the firmware
        // INT 16h queue. Before the OS owns the keyboard, bootfix.bin reads the
        // BIOS queue rather than the i8042 and would otherwise ignore this input.
        // `AERO_INJECT_KEY=Enter` (or Space, Escape, Tab). Re-injected every ~50
        // host slices so press+release can wake setup/LogonUI paint without a
        // synthetic FORCE_RESUME fill. Idle HLT snapshots must not use the
        // repeating path (each 0-inst slice still counts): set
        // `AERO_INJECT_KEY_ONCE=1` for a single press+release.
        let inject_key = std::env::var("AERO_INJECT_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let inject_key_once = std::env::var_os("AERO_INJECT_KEY_ONCE").is_some();
        if let Some(ref code) = inject_key {
            machine.inject_browser_key(code, true);
            machine.inject_browser_key(code, false);
            if let Some(key) = browser_code_to_bios_key(code) {
                machine.inject_bios_key(key);
            }
            if inject_key_once {
                eprintln!("AERO_INJECT_KEY: one-shot press+release '{code}' (no re-inject)");
            } else {
                eprintln!("AERO_INJECT_KEY: initial press+release '{code}'");
            }
        }
        // One left-click at the current guest cursor. The leftover Install-now
        // splash is a custom control that ignores Enter; the cursor already sits
        // on that button. Must not repeat on idle HLT slices.
        let inject_click_once = std::env::var_os("AERO_INJECT_CLICK_ONCE").is_some();
        if inject_click_once {
            machine.inject_mouse_left(true);
            machine.inject_mouse_left(false);
            eprintln!("AERO_INJECT_CLICK_ONCE: left press+release at current cursor");
        }
        // One-shot relative mouse: AERO_INJECT_MOUSE_ONCE=dx,dy[,click]
        // dy positive is down. Optional third field 1 = left-click after the move.
        if let Ok(spec) = std::env::var("AERO_INJECT_MOUSE_ONCE") {
            let mut parts = spec.split(',');
            let dx = parts
                .next()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let dy = parts
                .next()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let click = parts.next().is_some_and(|s| s == "1");
            machine.inject_mouse_motion(dx, dy, 0);
            if click {
                machine.inject_mouse_left(true);
                machine.inject_mouse_left(false);
            }
            eprintln!("AERO_INJECT_MOUSE_ONCE: dx={dx} dy={dy} click={click}");
        }
        let mut inject_key_slice: u64 = 0;

        let start = Instant::now();
        let mut total_executed: u64 = 0;
        let mut ir_jit = IrBlockCache::new();
        let mut run_error: Option<anyhow::Error> = None;
        let mut last_vga_dump = Instant::now();
        let mut last_progress = Instant::now();
        // Consecutive `run_slice` returns that executed 0 instructions while the guest was
        // interruptibly halted. Each such slice advances ~1ms of platform time inside the
        // machine; without a cap, max-insts runs would spin forever once Windows hits idle HLT.
        let mut idle_halt_slices: u64 = 0;
        let mut logged_interruptible_halt = false;
        let log_new_procs = std::env::var_os("AERO_LOG_NEW_PROCS").is_some();
        let new_proc_every: u64 = std::env::var("AERO_LOG_NEW_PROCS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(10_000_000);
        let mut seen_proc_keys: std::collections::BTreeSet<(u64, [u8; 15])> =
            std::collections::BTreeSet::new();
        let mut last_new_proc_scan: u64 = 0;
        if log_new_procs {
            eprintln!(
                "AERO_LOG_NEW_PROCS: walking ActiveProcessLinks every {new_proc_every} retired instructions"
            );
            for (pid, name, _eprocess) in walk_active_process_image_names(&mut machine) {
                let end = name.iter().position(|&b| b == 0).unwrap_or(15);
                let nm = String::from_utf8_lossy(&name[..end]);
                eprintln!("AERO_LOG_NEW_PROCS: inst=0 pid={pid} name='{nm}' (seed)");
                seen_proc_keys.insert((pid, name));
            }
        }

        loop {
            // Sticky IF: HAL clock init does CLI then wait-for-RTC-ISR; without IF the
            // 0x5C/0x10B RDTSC deadline always wins. Re-arm every host slice.
            if force_if {
                machine
                    .cpu_core_mut_by_index(0)
                    .state
                    .set_flag(aero_cpu_core::state::RFLAGS_IF, true);
            }
            if let Some(v) = force_cr8 {
                machine.cpu_core_mut_by_index(0).state.control.cr8 = v;
            }
            if let Some(ref code) = inject_key {
                if inject_key_repeats_each_slice(inject_key_once) {
                    inject_key_slice = inject_key_slice.wrapping_add(1);
                    // Every ~50 slices (~5M insts): press then release on next slice.
                    if inject_key_slice % 50 == 1 {
                        machine.inject_browser_key(code, true);
                        if let Some(key) = browser_code_to_bios_key(code) {
                            machine.inject_bios_key(key);
                        }
                    } else if inject_key_slice % 50 == 2 {
                        machine.inject_browser_key(code, false);
                    }
                }
            }

            // Sticky linear poke: re-apply every slice so kernel CR3 switches keep the
            // intended bytes visible (one-shot poke under winload was a no-op for HAL).
            if let Some(spec) = poke_spec.as_deref() {
                let cr3 = machine.cpu().control.cr3;
                if let Ok(msg) = poke_linear_regions(&mut machine, spec) {
                    // Log first success only — CR3 thrashing would otherwise flood stderr.
                    if !poke_logged_ok {
                        eprintln!("AERO_POKE_LINEAR (sticky):{msg} cr3={cr3:#x}");
                        poke_logged_ok = true;
                        poke_last_cr3 = Some(cr3);
                    } else {
                        let _ = (msg, poke_last_cr3);
                    }
                }
            }
            // Sticky multi-entry type-5 HOOK livelock wedge (re-scan each slice).
            if auto_type5 {
                let _ = wedge_type5_handle_table(&mut machine);
            }
            // Sticky WpeUtil IAT → ret-0 (skip SCE wait once winpeshl is mapped).
            if auto_sce {
                let _ = wedge_sce_wpeutil_iat(&mut machine);
            }
            // Sticky LFB identity map under the *current* CR3 (re-apply after switches).
            if auto_map_lfb {
                let _ = map_vbe_lfb_identity_2m(&mut machine);
            }
            for &va in &map_lfb_vas {
                let _ = map_va_to_vbe_lfb_2m(&mut machine, va);
            }
            if auto_redir_eng {
                if let Ok((_, eng_pa, surf_pa)) =
                    redir_eng_lfb_to_bootvid_kva(&mut machine, redir_eng_targets)
                {
                    redir_eng_targets = Some((eng_pa, surf_pa));
                }
            }

            let exit = if let Some(max_insts) = args.max_insts {
                if total_executed >= max_insts {
                    break;
                }
                let budget = (max_insts - total_executed).min(SLICE_INST_BUDGET);
                if let Some(tier) = compiled_tier.as_mut() {
                    machine.run_slice_compiled(budget, tier)
                } else if args.jit {
                    machine.run_slice_jit(budget, &mut ir_jit)
                } else {
                    machine.run_slice(budget)
                }
            } else {
                let max_ms = args
                    .max_ms
                    .expect("clap enforces that one of max_insts/max_ms is set");
                if start.elapsed() >= Duration::from_millis(max_ms) {
                    break;
                }
                if let Some(tier) = compiled_tier.as_mut() {
                    machine.run_slice_compiled(SLICE_INST_BUDGET, tier)
                } else if args.jit {
                    machine.run_slice_jit(SLICE_INST_BUDGET, &mut ir_jit)
                } else {
                    machine.run_slice(SLICE_INST_BUDGET)
                }
            };

            let slice_executed = exit.executed();
            total_executed = total_executed.saturating_add(slice_executed);
            stream_serial(&mut machine, &mut serial_sink)?;
            if let Some(out) = debugcon_sink.as_mut() {
                stream_debugcon(&mut machine, out)?;
            }

            // Periodic framebuffer dump so long boots can be watched as they progress.
            if let (Some(path), Some(interval)) = (&args.vga_png, args.vga_png_interval) {
                if last_vga_dump.elapsed() >= Duration::from_millis(interval) {
                    // The framebuffer may not be valid yet (e.g. text mode / no mode set);
                    // ignore dump errors here — the final dump on exit reports them properly.
                    let _ = dump_vga_png(&mut machine, path);
                    last_vga_dump = Instant::now();
                }
            }

            if log_new_procs && total_executed.saturating_sub(last_new_proc_scan) >= new_proc_every
            {
                last_new_proc_scan = total_executed;
                let census = walk_active_process_image_names(&mut machine);
                for (pid, name, _eprocess) in &census {
                    if seen_proc_keys.insert((*pid, *name)) {
                        let end = name.iter().position(|&b| b == 0).unwrap_or(15);
                        let nm = String::from_utf8_lossy(&name[..end]);
                        eprintln!(
                            "AERO_LOG_NEW_PROCS: inst={total_executed} pid={pid} name='{nm}'"
                        );
                    }
                }
            }

            if let Some(interval) = args.progress_interval {
                if last_progress.elapsed() >= Duration::from_millis(interval) {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 {
                        (total_executed as f64) / elapsed / 1e6
                    } else {
                        0.0
                    };
                    let mode = machine.cpu().mode;
                    let rip = machine.cpu().rip();
                    let cr3 = machine.cpu().control.cr3;
                    let cr8 = machine.cpu().control.cr8;
                    let halted = machine.cpu().halted;
                    let if_flag = (machine.cpu().rflags() & (1u64 << 9)) != 0;
                    machine.display_present();
                    let (dw, dh) = machine.display_resolution();
                    let fb = machine.display_framebuffer();
                    let non_zero = fb.iter().filter(|&&p| p != 0 && p != 0xFF00_0000).count();
                    let lfb_wr = machine.lfb_write_count();
                    eprintln!(
                        "[progress] t={elapsed:.1}s inst={total_executed} ({rate:.2} Minst/s) mode={mode:?} rip={rip:#x} cr3={cr3:#x} cr8={cr8:#x} if={if_flag} halted={halted} display={dw}x{dh} non_zero={non_zero} lfb_wr={lfb_wr}"
                    );
                    last_progress = Instant::now();
                }
            }

            // Pure interruptible idle: each Halted+0-exec slice advances ~1ms platform time.
            // Cap so max-insts runs cannot hang the host forever when timers never wake work.
            const MAX_IDLE_HALT_SLICES: u64 = 120_000; // ~120s platform time
            if matches!(exit, RunExit::Halted { .. }) && slice_executed == 0 {
                idle_halt_slices = idle_halt_slices.saturating_add(1);
                if idle_halt_slices >= MAX_IDLE_HALT_SLICES {
                    eprintln!(
                        "stopping after prolonged interruptible idle (~{idle_halt_slices} ms platform, {total_executed} inst)"
                    );
                    break;
                }
            } else {
                idle_halt_slices = 0;
            }

            match handle_exit(
                &mut machine,
                exit,
                total_executed,
                &mut cd_first_enabled,
                &mut logged_interruptible_halt,
            ) {
                Ok(LoopControl::Continue) => continue,
                Ok(LoopControl::Break) => break,
                Err(e) => {
                    run_error = Some(e);
                    break;
                }
            }
        }

        if std::env::var_os("AERO_FINAL_STATE").is_some() {
            let s = machine.cpu();
            eprintln!(
                "AERO_FINAL: executed={total_executed} rip={:#x} rsp={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} cr3={:#x}",
                s.rip(),
                s.read_gpr64(aero_cpu_core::state::gpr::RSP),
                s.read_gpr64(aero_cpu_core::state::gpr::RAX),
                s.read_gpr64(aero_cpu_core::state::gpr::RBX),
                s.read_gpr64(aero_cpu_core::state::gpr::RCX),
                s.read_gpr64(aero_cpu_core::state::gpr::RDX),
                s.control.cr3,
            );
        }
        if let Some(tier) = compiled_tier.as_ref() {
            eprintln!(
                "AERO_CT_STATS: compiled_blocks={} executed_blocks={} executed_guest_instructions={}",
                tier.compiled_blocks(),
                tier.executed_blocks(),
                tier.executed_guest_instructions()
            );
            eprintln!(
                "AERO_CT_CHAINS: {} chains, {:.2} blocks each, {} with no budget to chain",
                tier.chains(),
                if tier.chains() == 0 {
                    0.0
                } else {
                    tier.executed_blocks() as f64 / tier.chains() as f64
                },
                tier.chains_no_budget()
            );
            if let Some((both, to_promoted, total)) = tier.promoted_edge_share() {
                eprintln!(
                    "AERO_CT_PROMOTED_EDGES: {:.1}% of steps land on a promoted block, \
                     {:.1}% run promoted-to-promoted (of {total} recorded)",
                    100.0 * to_promoted as f64 / total as f64,
                    100.0 * both as f64 / total as f64
                );
            }
            let (hits, misses) = tier.module_cache_stats();
            if hits + misses > 0 {
                eprintln!(
                    "AERO_CT_MODULE_CACHE: {hits} modules loaded, {misses} compiled ({:.1}% reused)",
                    100.0 * hits as f64 / (hits + misses) as f64
                );
            }
            {
                let (h, m) = tier.promote_cache_stats();
                if h + m > 0 {
                    eprintln!(
                        "AERO_CT_PROMOTE_CACHE: {h} promoted modules loaded, {m} compiled ({:.1}% reused)",
                        100.0 * h as f64 / (h + m) as f64
                    );
                }
            }
            {
                let misses = aero_jit_x86::backend::JIT_TLB_MISSES
                    .load(std::sync::atomic::Ordering::Relaxed);
                let flushes = aero_jit_x86::backend::JIT_TLB_FLUSHES
                    .load(std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "AERO_CT_TLB: {misses} misses ({:.3}/guest insn), {flushes} full flushes \
                     (one per {:.0} guest insns, {:.1} misses each)",
                    misses as f64 / tier.executed_guest_instructions().max(1) as f64,
                    tier.executed_guest_instructions() as f64 / flushes.max(1) as f64,
                    misses as f64 / flushes.max(1) as f64
                );
            }
            {
                let k = tier.end_kinds();
                let total: u64 = k.iter().sum();
                eprintln!(
                    "AERO_CT_END_KIND: jmp={} jcc={} call={} ret={} exit={} limit={} \
                     (jmp is {:.1}% of {total} admitted)",
                    k[0],
                    k[1],
                    k[2],
                    k[3],
                    k[4],
                    k[5],
                    100.0 * k[0] as f64 / total.max(1) as f64
                );
            }
            if tier.jmp_block_steps() > 0 {
                eprintln!(
                    "AERO_CT_JMP_WEIGHT: {} of {} recorded chain steps enter a block \
                     that ends at an unconditional jump ({:.1}%)",
                    tier.jmp_block_steps(),
                    tier.recorded_chain_steps(),
                    100.0 * tier.jmp_block_steps() as f64 / tier.recorded_chain_steps().max(1) as f64
                );
            }
            let chained = tier.wasm_chain_steps();
            eprintln!(
                "AERO_CT_WASM_CHAIN: {chained} of {} block entries taken inside wasm ({:.1}%)",
                tier.executed_blocks(),
                if tier.executed_blocks() == 0 {
                    0.0
                } else {
                    100.0 * chained as f64 / tier.executed_blocks() as f64
                }
            );
            eprintln!(
                "AERO_CT_COST: {:.1} ns per compiled guest insn, {:.1} us per block admitted",
                tier.ns_per_executed_insn(),
                tier.us_per_compile()
            );
            eprintln!(
                "AERO_CT_FLAGS: {} dead flag writes dropped",
                tier.flag_writes_dropped()
            );
            let (empty, bails, cap, misses) = tier.rejection_counts();
            eprintln!(
                "AERO_CT_REJECT: empty={empty} bails_to_interp={bails} at_cap={cap} slow_mem={} chain_misses={misses}",
                tier.rejected_slow_mem()
            );
            let (edge_total, edge_top, edge_blocks) = tier.edge_concentration();
            if edge_total > 0 {
                eprintln!(
                    "AERO_CT_EDGES: {edge_total} transitions from {edge_blocks} blocks; {edge_top} ({:.1}%) went to the block's most frequent successor",
                    100.0 * edge_top as f64 / edge_total as f64
                );
            }
            if edge_total > 0 {
                let (hot, backward) = tier.hot_edge_direction();
                eprintln!(
                    "AERO_CT_EDGE_DIR: of {hot} hot-edge transitions, {backward} ({:.1}%) go backwards",
                    100.0 * backward as f64 / hot.max(1) as f64
                );
                let (cond_hot, ft) = tier.hot_edge_is_fallthrough();
                if cond_hot > 0 {
                    eprintln!(
                        "AERO_CT_EDGE_FT: of {cond_hot} hot-edge transitions out of conditional blocks, {ft} ({:.1}%) are the fall-through",
                        100.0 * ft as f64 / cond_hot as f64
                    );
                }
            }
            let (offered, dev, dun) = tier.seam_counts();
            eprintln!(
                "AERO_CT_SEAM: offered={offered} declined_event={dev} declined_uncompiled={dun} ran={}",
                offered - dev - dun
            );
            let (rejected_offers, busiest) = tier.rejected_offer_counts();
            // Counted over seam offers *and* chain steps — every time execution
            // wanted this block and could not have it — so it is not a subset of
            // the seam's decline count.
            eprintln!("AERO_CT_WALL: {rejected_offers} wanted executions of blocks Tier-1 refused");
            if let Some((total, blocks, shares)) = tier.execution_concentration() {
                eprintln!("AERO_CT_HOTNESS: {total} chain-step executions across {blocks} blocks");
                for (take, pct) in shares {
                    eprintln!("AERO_CT_HOTNESS_TOP: busiest {take} blocks cover {pct:.1}%");
                }
            }
            eprintln!(
                "AERO_CT_FOOTPRINT: {:.1} MiB of WASM emitted for {} blocks ({} bytes each)",
                tier.wasm_bytes() as f64 / (1024.0 * 1024.0),
                tier.compiled_blocks(),
                tier.wasm_bytes() / tier.compiled_blocks().max(1)
            );
            for (bytes, n) in aero_machine::compiled_tier::assist_histogram() {
                eprintln!("AERO_CT_ASSIST_TOP: {bytes} x{n}");
            }
            eprintln!(
                "AERO_CT_ASSISTS: {} instructions performed by the host inside a block, in {} blocks",
                tier.assists(),
                tier.blocks_with_assists()
            );
            if tier.stepped_over() > 0 {
                eprintln!(
                    "AERO_CT_STEPPED: {} guest instructions stepped over to get past a refusal",
                    tier.stepped_over()
                );
            }
            let (promotions, us) = tier.promotion_counts();
            if promotions > 0 {
                eprintln!(
                    "AERO_CT_TIERUP: {promotions} blocks recompiled by Cranelift, {us:.1} us each"
                );
            }
            let (sites, hits) = tier.resume_counts();
            if sites > 0 {
                eprintln!(
                    "AERO_CT_RESUME: {sites} resume points armed, {hits} reached ({:.1}%)",
                    100.0 * hits as f64 / sites as f64
                );
            }
            let (steps, same) = tier.chain_locality();
            if steps > 0 {
                eprintln!(
                    "AERO_CT_LOCALITY: {same} of {steps} chain steps ({:.1}%) stay inside one module",
                    100.0 * same as f64 / steps as f64
                );
            }
            for (bytes, n) in busiest {
                eprintln!("AERO_CT_WALL_TOP: {bytes} x{n}");
            }
            eprintln!(
                "AERO_CT_LOOPS: self_loop_steps={} of {} block executions",
                tier.self_loop_steps(),
                tier.executed_blocks()
            );
            for (name, n) in tier.bail_mnemonics().into_iter().take(12) {
                eprintln!("AERO_CT_BAIL: {name} x{n}");
            }
        }
        if args.jit {
            eprintln!(
                "AERO_JIT_STATS: cached_blocks={} executed_blocks={} executed_guest_instructions={} memory_free_blocks={} memory_free_insts={}",
                ir_jit.len(),
                ir_jit.executed_blocks(),
                ir_jit.executed_guest_instructions(),
                ir_jit.memory_free_blocks(),
                ir_jit.memory_free_insts()
            );
        }

        // Flush any remaining serial bytes.
        stream_serial(&mut machine, &mut serial_sink)?;
        if let Err(e) = serial_sink.flush() {
            if run_error.is_some() {
                eprintln!("warning: failed to flush serial output: {e}");
            } else {
                return Err(e.into());
            }
        }
        if let Some(out) = debugcon_sink.as_mut() {
            stream_debugcon(&mut machine, out)?;
            if let Err(e) = out.flush() {
                if run_error.is_some() {
                    eprintln!("warning: failed to flush debugcon output: {e}");
                } else {
                    return Err(e.into());
                }
            }
        }

        if let Some(path) = &args.snapshot_save {
            let mut f = File::create(path).with_context(|| {
                format!(
                    "failed to create snapshot file for save: {}",
                    path.display()
                )
            })?;
            if let Err(e) = machine
                .save_snapshot_full_to(&mut f)
                .map_err(|e| anyhow!("{e}"))
            {
                if run_error.is_some() {
                    eprintln!(
                        "warning: failed to save snapshot to {}: {e}",
                        path.display()
                    );
                } else {
                    return Err(e);
                }
            }
        }

        if let Some(path) = &args.vga_png {
            if let Err(e) = dump_vga_png(&mut machine, path) {
                if run_error.is_some() {
                    eprintln!("warning: failed to dump VGA PNG to {}: {e}", path.display());
                } else {
                    return Err(e);
                }
            }
        }

        // Clean-stop memory probe: AERO_DUMP_MEM also fires on non-fault exits (error
        // screens, halts, cap reached) — the fault path prints it via the exception
        // Bring-up: dump CR2/CR8/GS bases on any stop (clean or fault).
        if std::env::var_os("AERO_DUMP_CPU").is_some() {
            let cpu = machine.cpu();
            eprintln!(
                "AERO_DUMP_CPU: cr0={:#x} cr2={:#x} cr3={:#x} cr4={:#x} cr8={:#x} efer={:#x} star={:#x} lstar={:#x} gs_base={:#x} kernel_gs_base={:#x} fs_base={:#x} rsp={:#x} rip={:#x} cs={:#x} ss={:#x} tr_base={:#x} tr_sel={:#x}",
                cpu.control.cr0,
                cpu.control.cr2,
                cpu.control.cr3,
                cpu.control.cr4,
                cpu.control.cr8,
                cpu.msr.efer,
                cpu.msr.star,
                cpu.msr.lstar,
                cpu.msr.gs_base,
                cpu.msr.kernel_gs_base,
                cpu.msr.fs_base,
                cpu.read_gpr64(aero_cpu_core::state::gpr::RSP),
                cpu.rip(),
                cpu.segments.cs.selector,
                cpu.segments.ss.selector,
                cpu.tables.tr.base,
                cpu.tables.tr.selector,
            );
        }
        // Full GPRs for handle-table / win32k bring-up (r8=index, r9=count mid-walk).
        if std::env::var_os("AERO_DUMP_GPRS").is_some() {
            use aero_cpu_core::state::gpr;
            let cpu = machine.cpu();
            eprintln!(
                "AERO_DUMP_GPRS: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rbp={:#x} rsp={:#x} r8={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x} rip={:#x} rflags={:#x}",
                cpu.read_gpr64(gpr::RAX),
                cpu.read_gpr64(gpr::RBX),
                cpu.read_gpr64(gpr::RCX),
                cpu.read_gpr64(gpr::RDX),
                cpu.read_gpr64(gpr::RSI),
                cpu.read_gpr64(gpr::RDI),
                cpu.read_gpr64(gpr::RBP),
                cpu.read_gpr64(gpr::RSP),
                cpu.read_gpr64(gpr::R8),
                cpu.read_gpr64(gpr::R9),
                cpu.read_gpr64(gpr::R10),
                cpu.read_gpr64(gpr::R11),
                cpu.read_gpr64(gpr::R12),
                cpu.read_gpr64(gpr::R13),
                cpu.read_gpr64(gpr::R14),
                cpu.read_gpr64(gpr::R15),
                cpu.rip(),
                cpu.rflags_snapshot(),
            );
        }
        // Display / VBE bring-up: mode, LFB base, and whether any non-zero pixels exist.
        if std::env::var_os("AERO_DUMP_VGA").is_some() {
            let path = machine.display_present_path();
            let (dw, dh) = machine.display_resolution();
            let fb = machine.display_framebuffer();
            let non_zero = fb.iter().filter(|&&p| p != 0 && p != 0xFF00_0000).count();
            let mode = machine.bios_video_mode();
            let vbe_mode = machine.vbe_current_mode();
            let lfb = machine.vbe_lfb_base();
            let (dispi_en, dispi_w, dispi_h, dispi_bpp) = machine.aerogpu_vbe_dispi();
            let (virt_w, virt_h, x_off, y_off, pitch) =
                machine.aerogpu_vbe_dispi_virtual_geometry();
            let (wddm, wddm_en, sw, sh, fmt, gpa) = machine.aerogpu_scanout0_debug();
            let lfb_writes = machine.lfb_write_count();
            eprintln!(
                "AERO_DUMP_VGA: path={path} display={dw}x{dh} fb_len={} non_zero_px={non_zero} bios_mode={mode:#x} vbe_mode={vbe_mode:?} lfb={lfb:#x} dispi_en={dispi_en:#x} dispi={dispi_w}x{dispi_h}x{dispi_bpp} virtual={virt_w}x{virt_h}+{x_off}+{y_off} pitch={pitch} wddm_active={wddm} wddm_en={wddm_en} scanout={sw}x{sh} fmt={fmt:#x} fb_gpa={gpa:#x} lfb_writes={lfb_writes}",
                fb.len()
            );
            let mut samples = Vec::new();
            for (i, &p) in fb.iter().enumerate() {
                if p != 0 && p != 0xFF00_0000 {
                    samples.push((i, p));
                    if samples.len() >= 4 {
                        break;
                    }
                }
            }
            if !samples.is_empty() {
                eprintln!("AERO_DUMP_VGA samples: {samples:?}");
            }
        }
        if std::env::var_os("AERO_DUMP_INPUT").is_some() {
            let ps2_state = machine.ps2_diagnostic_state();
            eprintln!(
                "AERO_DUMP_INPUT: ps2_present={} ps2_leds={:#04x} usb_keyboard_configured={} \
                 usb_keyboard_leds={:#04x} virtio_keyboard_driver_ok={} virtio_keyboard_leds={:#04x} \
                 ps2_state={ps2_state:?}",
                machine.ps2_available(),
                machine.ps2_keyboard_leds(),
                machine.usb_hid_keyboard_configured(),
                machine.usb_hid_keyboard_leds(),
                machine.virtio_input_keyboard_driver_ok(),
                machine.virtio_input_keyboard_leds(),
            );
        }
        // Enumerate PCI bus 0 functions for bring-up (IDE/ATAPI/AHCI presence).
        if std::env::var_os("AERO_DUMP_PCI").is_some() {
            // Enumerate bus 0 via PCI config mechanism #1 (0xCF8/0xCFC) so we
            // do not need a direct aero-devices dependency in this binary.
            if let Some(pci_cfg) = machine.pci_config_ports() {
                let mut pci_cfg = pci_cfg.borrow_mut();
                let mut parts = String::new();
                for dev in 0u8..32 {
                    for func in 0u8..8 {
                        let addr = 0x8000_0000u32 | ((dev as u32) << 11) | ((func as u32) << 8);
                        pci_cfg.io_write(0xCF8, 4, addr);
                        let id = pci_cfg.io_read(0xCFC, 4);
                        let vendor = (id & 0xffff) as u16;
                        if vendor == 0xFFFF {
                            if func == 0 {
                                break; // no device at this slot
                            }
                            continue;
                        }
                        let device = ((id >> 16) & 0xffff) as u16;
                        pci_cfg.io_write(0xCF8, 4, addr | 0x04);
                        let cmd = (pci_cfg.io_read(0xCFC, 4) & 0xffff) as u16;
                        pci_cfg.io_write(0xCF8, 4, addr | 0x08);
                        let class_reg = pci_cfg.io_read(0xCFC, 4);
                        let rev = (class_reg & 0xff) as u8;
                        let prog = ((class_reg >> 8) & 0xff) as u8;
                        let sub = ((class_reg >> 16) & 0xff) as u8;
                        let base = ((class_reg >> 24) & 0xff) as u8;
                        pci_cfg.io_write(0xCF8, 4, addr | 0x0c);
                        let hdr = ((pci_cfg.io_read(0xCFC, 4) >> 16) & 0xff) as u8;
                        // BAR0/BAR1 matter for display/storage bind diagnosis.
                        pci_cfg.io_write(0xCF8, 4, addr | 0x10);
                        let bar0 = pci_cfg.io_read(0xCFC, 4);
                        pci_cfg.io_write(0xCF8, 4, addr | 0x14);
                        let bar1 = pci_cfg.io_read(0xCFC, 4);
                        parts.push_str(&format!(
                            " {dev:02x}.{func}:ven={vendor:04x} dev={device:04x} class={base:02x}/{sub:02x}/{prog:02x} rev={rev:02x} hdr={hdr:02x} cmd={cmd:04x} bar0={bar0:#x} bar1={bar1:#x}"
                        ));
                        if func == 0 && (hdr & 0x80) == 0 {
                            break; // single-function device
                        }
                    }
                }
                eprintln!("AERO_DUMP_PCI (bus0):{parts}");
            } else {
                eprintln!("AERO_DUMP_PCI: (no pci config ports)");
            }
        }
        // Per-vCPU LAPIC is not visible via AERO_DUMP_MEM (open-bus 0xFF on the shared
        // physical path). Dump the routed view used by the BSP.
        if std::env::var_os("AERO_DUMP_LAPIC").is_some() {
            let mut offs = vec![
                (0x20u64, "id"),
                (0x30, "ver"),
                (0x80, "tpr"),
                (0xa0, "ppr"),
                (0xd0, "ldr"),
                (0xe0, "dfr"),
                (0xf0, "svr"),
                (0x320, "lvt_timer"),
                (0x350, "lvt_lint0"),
                (0x360, "lvt_lint1"),
                (0x370, "lvt_error"),
                (0x380, "init_count"),
                (0x390, "cur_count"),
                (0x3e0, "divide"),
            ];
            const LAPIC_BITMAP_BANKS: [(u64, &str); 3] =
                [(0x100, "isr"), (0x200, "tmr"), (0x280, "irr")];
            for (base, name) in LAPIC_BITMAP_BANKS {
                for bank in 0u64..8 {
                    offs.push((
                        base + bank * 0x10,
                        match (name, bank) {
                            ("isr", 0) => "isr0",
                            ("isr", 1) => "isr1",
                            ("isr", 2) => "isr2",
                            ("isr", 3) => "isr3",
                            ("isr", 4) => "isr4",
                            ("isr", 5) => "isr5",
                            ("isr", 6) => "isr6",
                            ("isr", 7) => "isr7",
                            ("tmr", 0) => "tmr0",
                            ("tmr", 1) => "tmr1",
                            ("tmr", 2) => "tmr2",
                            ("tmr", 3) => "tmr3",
                            ("tmr", 4) => "tmr4",
                            ("tmr", 5) => "tmr5",
                            ("tmr", 6) => "tmr6",
                            ("tmr", 7) => "tmr7",
                            ("irr", 0) => "irr0",
                            ("irr", 1) => "irr1",
                            ("irr", 2) => "irr2",
                            ("irr", 3) => "irr3",
                            ("irr", 4) => "irr4",
                            ("irr", 5) => "irr5",
                            ("irr", 6) => "irr6",
                            ("irr", 7) => "irr7",
                            _ => unreachable!(),
                        },
                    ));
                }
            }
            let mut parts = String::new();
            for (off, name) in offs {
                let v = machine.read_lapic_u32(0, off);
                parts.push_str(&format!(" {name}[{off:#x}]={v:#x}"));
            }
            eprintln!("AERO_DUMP_LAPIC (cpu0):{parts}");
        }
        // All 24 Q35 IOAPIC redirections, including the PCI APIC window at GSIs 16..23.
        if std::env::var_os("AERO_DUMP_IOAPIC").is_some() {
            if let Some(interrupts) = machine.platform_interrupts() {
                let mut parts = String::new();
                let mut ints = interrupts.borrow_mut();
                for gsi in 0u32..24 {
                    let low_idx = 0x10 + gsi * 2;
                    let high_idx = low_idx + 1;
                    ints.ioapic_mmio_write(0x00, low_idx);
                    let low = ints.ioapic_mmio_read(0x10);
                    ints.ioapic_mmio_write(0x00, high_idx);
                    let high = ints.ioapic_mmio_read(0x10);
                    let masked = (low & (1 << 16)) != 0;
                    let vector = low & 0xff;
                    parts.push_str(&format!(
                        " gsi{gsi}:vec={vector:#x} mask={masked} low={low:#x} high={high:#x}"
                    ));
                }
                eprintln!("AERO_DUMP_IOAPIC:{parts}");
            } else {
                eprintln!("AERO_DUMP_IOAPIC: (no platform interrupts)");
            }
        }
        if std::env::var_os("AERO_DUMP_CPU_INTERNAL").is_some() {
            let core = machine.cpu_core_mut_by_index(0);
            let external = core
                .pending
                .external_interrupts()
                .iter()
                .map(|vector| format!("{vector:#x}"))
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "AERO_DUMP_CPU_INTERNAL (cpu0): rip={:#x} rflags={:#x} cr8={:#x} halted={} tsc={:#x} interrupt_inhibit={} pending_event={} external=[{}] dropped_external={}",
                core.state.rip(),
                core.state.rflags(),
                core.state.control.cr8,
                core.state.halted,
                core.time.read_tsc(),
                core.pending.interrupt_inhibit(),
                core.pending.has_pending_event(),
                external,
                core.pending.dropped_external_interrupts(),
            );
        }

        // context, so skip when there is a run error to avoid a duplicate dump.
        if run_error.is_none() {
            if let Some(spec) = std::env::var("AERO_DUMP_MEM")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_DUMP_MEM (clean stop at {total_executed} inst):{}",
                    dump_mem_regions(&mut machine, &spec)
                );
            }
            // Linear (VA) dump for long-mode bring-up: AERO_DUMP_LINEAR=<hex_va>:<len>[,...]
            // Prints a page-table walk + bytes (unmapped → 0xFE).
            if let Some(spec) = std::env::var("AERO_DUMP_LINEAR")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_DUMP_LINEAR (clean stop at {total_executed} inst):{}",
                    dump_linear_regions(&mut machine, &spec)
                );
            }
            // Scan guest physical RAM for ASCII needles (bring-up: process names, etc.).
            // AERO_SCAN_ASCII=smss.exe,System,csrss.exe  (max 8 needles, first 16 hits each)
            if let Some(spec) = std::env::var("AERO_SCAN_ASCII")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_SCAN_ASCII (clean stop at {total_executed} inst):{}",
                    scan_ascii_in_ram(&mut machine, &spec)
                );
            }
            // UTF-16LE needles (Windows object names, registry paths, etc.).
            // AERO_SCAN_UTF16=\Device\Video0,winpeshl.exe  (max 8; encoded as UTF-16LE)
            if let Some(spec) = std::env::var("AERO_SCAN_UTF16")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_SCAN_UTF16 (clean stop at {total_executed} inst):{}",
                    scan_utf16_in_ram(&mut machine, &spec)
                );
            }
            // Little-endian u32/u64 needles in guest physical RAM (bring-up: LFB/BAR
            // phys bases, surface pointers). AERO_SCAN_U32=e1000000,e4040000
            // AERO_SCAN_U64=fffff7ffefd00000
            if let Some(spec) = std::env::var("AERO_SCAN_U32")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_SCAN_U32 (clean stop at {total_executed} inst):{}",
                    scan_le_int_in_ram(&mut machine, &spec, 4)
                );
            }
            if let Some(spec) = std::env::var("AERO_SCAN_U64")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_SCAN_U64 (clean stop at {total_executed} inst):{}",
                    scan_le_int_in_ram(&mut machine, &spec, 8)
                );
            }
            // Raw little-endian byte needles (hex, no spaces). Bring-up: SURFOBJ
            // sizlBitmap+cjBits = 1024,768,0x300000 → 000400000003000000003000
            // AERO_SCAN_BYTES=000400000003000000003000
            if let Some(spec) = std::env::var("AERO_SCAN_BYTES")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_SCAN_BYTES (clean stop at {total_executed} inst):{}",
                    scan_bytes_in_ram(&mut machine, &spec)
                );
            }
            // Heuristic SURFOBJ walk: 1024×768×32 primary candidates + pvBits.
            if std::env::var_os("AERO_SCAN_SURFOBJ").is_some() {
                eprintln!(
                    "AERO_SCAN_SURFOBJ (clean stop at {total_executed} inst):{}",
                    scan_surfobj_candidates(&mut machine)
                );
            }
            // Heuristic Win7 SP1 x64 EPROCESS walk via ImageFileName @ +0x2e0.
            if std::env::var_os("AERO_DUMP_PROCS").is_some()
                || std::env::var_os("AERO_DUMP_THREADS").is_some()
                || std::env::var_os("AERO_DUMP_MODULES").is_some()
            {
                eprintln!(
                    "AERO_DUMP_PROCS (clean stop at {total_executed} inst):{}",
                    dump_eprocess_image_names(&mut machine)
                );
            }
            // Symbol-derived Win7 SP1 x64 PnP devnode walk. The value is a
            // comma-separated list of case-insensitive InstancePath filters.
            if let Some(spec) = std::env::var("AERO_DUMP_DEVNODES")
                .ok()
                .filter(|s| !s.is_empty())
            {
                eprintln!(
                    "AERO_DUMP_DEVNODES (clean stop at {total_executed} inst):{}",
                    dump_win7_devnodes(&mut machine, &spec)
                );
            }
        }

        if let Some(e) = run_error {
            return Err(e);
        }

        Ok(())
    }

    fn open_disk_image(path: &Path, read_only: bool) -> Result<DiskImage<StdFileBackend>> {
        let backend = if read_only {
            StdFileBackend::open_read_only(path)
        } else {
            StdFileBackend::open_rw(path)
        }
        .map_err(|e| anyhow!("failed to open disk image {}: {e}", path.display()))?;

        let disk = DiskImage::open_auto(backend)
            .map_err(|e| anyhow!("failed to open disk image {}: {e}", path.display()))?;

        let capacity = disk.capacity_bytes();
        if capacity == 0 {
            bail!(
                "disk image is empty (expected at least one {}-byte sector)",
                SECTOR_SIZE
            );
        }
        if capacity % SECTOR_SIZE as u64 != 0 {
            bail!(
                "disk image capacity {} is not a multiple of {} bytes",
                capacity,
                SECTOR_SIZE
            );
        }

        Ok(disk)
    }

    fn open_disk_backend(path: &Path, read_only: bool) -> Result<Box<dyn VirtualDisk>> {
        let disk = open_disk_image(path, read_only)?;
        Ok(Box::new(disk))
    }

    fn open_disk_backend_with_overlay(
        base_path: &Path,
        overlay_path: &Path,
        create_block_size: u32,
    ) -> Result<Box<dyn VirtualDisk>> {
        let base = open_disk_image(base_path, true)?;

        let overlay_exists = overlay_path.exists();
        let overlay_backend = if overlay_exists {
            StdFileBackend::open_rw(overlay_path)
        } else {
            StdFileBackend::create(overlay_path, 0)
        }
        .map_err(|e| {
            anyhow!(
                "failed to open overlay image {}: {e}",
                overlay_path.display()
            )
        })?;

        let cow = if overlay_exists {
            AeroCowDisk::open(base, overlay_backend)
        } else {
            AeroCowDisk::create(base, overlay_backend, create_block_size)
        }
        .map_err(|e| anyhow!("failed to initialize COW overlay disk: {e}"))?;

        Ok(Box::new(cow))
    }

    fn open_serial_sink(serial_out: &str) -> Result<Box<dyn Write>> {
        if serial_out == "stdout" {
            return Ok(Box::new(io::stdout()));
        }
        let f = File::create(serial_out)
            .with_context(|| format!("failed to create serial output file: {serial_out}"))?;
        Ok(Box::new(BufWriter::new(f)))
    }

    fn open_optional_sink(dest: &str) -> Result<Option<Box<dyn Write>>> {
        if dest == "none" {
            return Ok(None);
        }
        Ok(Some(open_serial_sink(dest)?))
    }

    fn stream_serial(machine: &mut Machine, out: &mut dyn Write) -> Result<()> {
        let bytes = machine.take_serial_output();
        if !bytes.is_empty() {
            out.write_all(&bytes)?;
        }
        Ok(())
    }

    fn stream_debugcon(machine: &mut Machine, out: &mut dyn Write) -> Result<()> {
        let bytes = machine.take_debugcon_output();
        if !bytes.is_empty() {
            out.write_all(&bytes)?;
        }
        Ok(())
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum LoopControl {
        Continue,
        Break,
    }

    fn rd64_phys(m: &mut Machine, pa: u64) -> u64 {
        let mut v = 0u64;
        for j in 0..8u64 {
            v |= u64::from(m.read_physical_u8(pa.wrapping_add(j))) << (j * 8);
        }
        v
    }

    /// Walk the 4-level long-mode page tables for `va` from `cr3`, returning a
    /// human-readable description of each level's PTE (or where the walk stops).
    /// Used to diagnose paging/VA-mapping divergences (e.g. whether the kernel is
    /// mapped high-half or low).
    fn walk_pt(machine: &mut Machine, cr3: u64, va: u64) -> String {
        let mut out = String::new();
        let mut table = cr3 & 0x000F_FFFF_FFFF_F000;
        let names = ["PML4", "PDPT", "PD", "PT"];
        for level in 0..4 {
            let idx = (va >> (39 - level * 9)) & 0x1FF;
            let pte = rd64_phys(machine, table.wrapping_add(idx * 8));
            let present = pte & 1 != 0;
            let huge = pte & 0x80 != 0;
            out.push_str(&format!(
                " {}[{}]={pte:#x}{}",
                names[level as usize],
                idx,
                if !present {
                    "(NP)"
                } else if huge {
                    "(HUGE)"
                } else {
                    ""
                }
            ));
            if !present || (huge && level >= 1) || level == 3 {
                return out;
            }
            table = pte & 0x000F_FFFF_FFFF_F000;
        }
        out
    }

    /// Translate a long-mode canonical VA via the guest's own page tables.
    /// Returns `None` if any level is not present. Large pages (1G/2M) are handled.
    fn translate_long(machine: &mut Machine, cr3: u64, va: u64) -> Option<u64> {
        let mut table = cr3 & 0x000F_FFFF_FFFF_F000;
        for level in 0..4 {
            let idx = (va >> (39 - level * 9)) & 0x1FF;
            let pte = rd64_phys(machine, table.wrapping_add(idx * 8));
            if pte & 1 == 0 {
                return None;
            }
            let huge = pte & 0x80 != 0;
            if huge && level == 1 {
                // 1 GiB page
                let base = pte & 0x000F_FFFF_C000_0000;
                return Some(base | (va & 0x3FFF_FFFF));
            }
            if huge && level == 2 {
                // 2 MiB page
                let base = pte & 0x000F_FFFF_FFE0_0000;
                return Some(base | (va & 0x001F_FFFF));
            }
            if level == 3 {
                let base = pte & 0x000F_FFFF_FFFF_F000;
                return Some(base | (va & 0xFFF));
            }
            table = pte & 0x000F_FFFF_FFFF_F000;
        }
        None
    }

    /// Read `len` bytes at a guest linear address. In long mode with paging, walk the
    /// guest page tables; otherwise treat the address as physical (legacy dump path).
    /// Cross-page reads are supported. Unmapped bytes are filled with `0xFE` (distinct
    /// from open-bus `0xFF`) so a failed walk is obvious.
    fn read_linear_bytes(
        machine: &mut Machine,
        linear: u64,
        len: usize,
        long_mode: bool,
    ) -> Vec<u8> {
        let cr3 = machine.cpu().control.cr3;
        read_linear_bytes_cr3(machine, linear, len, long_mode, cr3)
    }

    fn read_linear_bytes_cr3(
        machine: &mut Machine,
        linear: u64,
        len: usize,
        long_mode: bool,
        cr3: u64,
    ) -> Vec<u8> {
        let mut out = vec![0u8; len];
        if !long_mode {
            for (i, b) in out.iter_mut().enumerate() {
                *b = machine.read_physical_u8(linear.wrapping_add(i as u64));
            }
            return out;
        }
        let mut i = 0usize;
        while i < len {
            let va = linear.wrapping_add(i as u64);
            match translate_long(machine, cr3, va) {
                Some(pa) => {
                    // Consume as much as remains in this page (4K granularity is enough;
                    // large pages still share the 4K offset progression).
                    let page_rem = 0x1000 - (pa & 0xFFF) as usize;
                    let chunk = page_rem.min(len - i);
                    for j in 0..chunk {
                        out[i + j] = machine.read_physical_u8(pa.wrapping_add(j as u64));
                    }
                    i += chunk;
                }
                None => {
                    out[i] = 0xFE;
                    i += 1;
                }
            }
        }
        out
    }

    fn exception_debug_context(machine: &mut Machine) -> String {
        let (rip, cs_sel, cs_base, mode) = {
            let cpu = machine.cpu();
            (
                cpu.rip(),
                cpu.segments.cs.selector,
                cpu.segments.cs.base,
                cpu.mode,
            )
        };
        let linear = cs_base.wrapping_add(rip);
        // Long mode (IA-32e) with paging: must walk guest PTs. Treating the linear
        // address as physical yields open-bus 0xFF and has already misled one
        // investigation (see wiki: the Windows 7 bring-up, unhandled MSRs).
        let long_mode = {
            let cpu = machine.cpu();
            let paging = (cpu.control.cr0 & (1 << 31)) != 0;
            let lma = (cpu.msr.efer & (1 << 10)) != 0; // EFER.LMA
            paging && lma
        };
        let _ = mode; // retained in the printed context below
        let bytes = read_linear_bytes(machine, linear, 16, long_mode);
        let mut hex = String::with_capacity(3 * 16);
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 {
                hex.push(' ');
            }
            hex.push_str(&format!("{b:02x}"));
        }
        // Backward bytes (rip-24..rip) for context on the preceding instructions.
        let back_bytes = read_linear_bytes(machine, linear.wrapping_sub(24), 24, long_mode);
        let mut back = String::new();
        for (i, b) in back_bytes.iter().enumerate() {
            if i > 0 {
                back.push(' ');
            }
            back.push_str(&format!("{b:02x}"));
        }
        // Also dump the stack top (SS:SP), which is where far-return selectors come from.
        let (ss_base, sp) = {
            let cpu = machine.cpu();
            (cpu.segments.ss.base, cpu.stack_ptr())
        };
        let stack_linear = ss_base.wrapping_add(sp);
        let stack_bytes = read_linear_bytes(machine, stack_linear, 16, long_mode);
        let mut stack_hex = String::new();
        for i in 0..8 {
            if i > 0 {
                stack_hex.push(' ');
            }
            let lo = stack_bytes[i * 2];
            let hi = stack_bytes[i * 2 + 1];
            let w = u16::from(lo) | (u16::from(hi) << 8);
            stack_hex.push_str(&format!("{w:04x}"));
        }
        // GDTR + the descriptor that a far transfer would fetch (error code's table index).
        let (gdtr_base, gdtr_limit) = {
            let cpu = machine.cpu();
            (cpu.tables.gdtr.base, cpu.tables.gdtr.limit)
        };
        let desc_index = 8u64;
        let desc_bytes = read_linear_bytes(
            machine,
            gdtr_base.wrapping_add(desc_index * 8),
            8,
            long_mode,
        );
        let mut desc_hex = String::new();
        for (i, b) in desc_bytes.iter().enumerate() {
            if i > 0 {
                desc_hex.push(' ');
            }
            desc_hex.push_str(&format!("{b:02x}"));
        }
        // Also dump the descriptor for the *current* CS selector (index = sel>>3), and the
        // dwords just below SP (the operands a RETF already consumed when it faulted).
        let cs_desc_index = (cs_sel >> 3) as u64;
        let cs_desc_bytes = read_linear_bytes(
            machine,
            gdtr_base.wrapping_add(cs_desc_index * 8),
            8,
            long_mode,
        );
        let mut cs_desc_hex = String::new();
        for (i, b) in cs_desc_bytes.iter().enumerate() {
            if i > 0 {
                cs_desc_hex.push(' ');
            }
            cs_desc_hex.push_str(&format!("{b:02x}"));
        }
        let below_bytes = read_linear_bytes(machine, stack_linear.wrapping_sub(8), 16, long_mode);
        let mut below_hex = String::new();
        for i in 0..4 {
            if i > 0 {
                below_hex.push(' ');
            }
            let mut v = 0u32;
            for j in 0..4 {
                v |= u32::from(below_bytes[i * 4 + j]) << (j * 8);
            }
            below_hex.push_str(&format!("{v:08x}"));
        }
        // Full GDT dump (limit+1)/8 entries, decoded.
        let entry_count = ((gdtr_limit as u64) + 1) / 8;
        let mut gdt_lines = String::new();
        for idx in 0..entry_count {
            let e = read_linear_bytes(machine, gdtr_base.wrapping_add(idx * 8), 8, long_mode);
            let limit = u32::from(e[0]) | (u32::from(e[1]) << 8) | ((u32::from(e[6]) & 0xF) << 16);
            let base = u32::from(e[2])
                | (u32::from(e[3]) << 8)
                | (u32::from(e[4]) << 16)
                | (u32::from(e[7]) << 24);
            let access = e[5];
            gdt_lines.push_str(&format!(
                " [{idx}] b={base:06x} l={limit:05x} a={access:02x}(t={:x},s={},d={})",
                access & 0xF,
                (access >> 4) & 1,
                (access >> 5) & 3
            ));
        }
        // Optional env-gated physical region dump: AERO_DUMP_MEM=<addr>:<len> (hex),
        // comma-separated for multiple regions. Useful for "data-as-code" faults (see
        // how far a zero/garbage region extends) and for inspecting firmware tables.
        let mut region_hex = String::new();
        if let Some(spec) = std::env::var("AERO_DUMP_MEM")
            .ok()
            .filter(|s| !s.is_empty())
        {
            region_hex = dump_mem_regions(machine, &spec);
        }
        // Page-table walks for the fault RIP and a high-half probe, to characterize the
        // kernel's VA mapping (paging/VA divergence diagnosis).
        let cr3 = machine.cpu().control.cr3;
        let pt_rip = walk_pt(machine, cr3, linear);
        let pt_high = walk_pt(machine, cr3, 0xFFFF_F880_0000_0000u64);
        let pt_hex = format!(" | cr3={cr3:#x} pt(rip)={pt_rip} pt(half)={pt_high}");
        format!(
            "rip={rip:#06x} cs.sel={cs_sel:#06x} cs.base={cs_base:#06x} linear={linear:#06x} mode={mode:?} bytes[-24]={back} | bytes={hex} | stack[ss:sp]={stack_hex} | gdtr={gdtr_base:#x}/{gdtr_limit:#x} desc[{desc_index}]={desc_hex} desc[{cs_desc_index}]={cs_desc_hex} | below_sp={below_hex} | gdt={gdt_lines}{region_hex}{pt_hex}"
        )
    }

    fn handle_exit(
        machine: &mut Machine,
        exit: RunExit,
        total_executed: u64,
        cd_first_enabled: &mut bool,
        logged_interruptible_halt: &mut bool,
    ) -> Result<LoopControl> {
        match exit {
            RunExit::Completed { .. } => Ok(LoopControl::Continue),
            RunExit::Halted { .. } => {
                // Windows (and any OS) uses HLT in the idle loop with IF=1. That is not a
                // terminal stop — the host must keep advancing platform time so PIT/RTC/LAPIC
                // can wake the guest. Only treat HLT as terminal when interrupts are masked
                // (classic firmware "cli; hlt" hang / end-of-fixture).
                let if_set = (machine.cpu().rflags() & aero_cpu_core::state::RFLAGS_IF) != 0;
                if if_set {
                    if !*logged_interruptible_halt {
                        eprintln!(
                            "guest interruptible HLT after {total_executed} instructions (continuing; timers may wake)"
                        );
                        *logged_interruptible_halt = true;
                    }
                    Ok(LoopControl::Continue)
                } else {
                    eprintln!("guest halted after {total_executed} instructions (IF=0; terminal)");
                    Ok(LoopControl::Break)
                }
            }
            RunExit::ResetRequested { kind, .. } => {
                // Bring-up: ignore CF9/i8042 resets so setup/winpeshl progress is not wiped
                // (observed: guest reset after winpeshl/new process → Real-mode BIOS).
                // AERO_IGNORE_RESET=1 keeps running without Machine::reset().
                let ignore = matches!(
                    std::env::var("AERO_IGNORE_RESET").as_deref(),
                    Ok("1") | Ok("true") | Ok("on") | Ok("yes")
                );
                if ignore {
                    eprintln!(
                        "guest requested reset: {kind:?} (AERO_IGNORE_RESET=1; NOT resetting)"
                    );
                    return Ok(LoopControl::Continue);
                }
                // When using the "CD-first when present" policy, Windows setup commonly boots from
                // CD once, then reboots into the installed HDD while leaving the ISO attached.
                // Disable the policy after the first guest reset so setup does not loop back into
                // install media.
                if *cd_first_enabled && machine.active_boot_device() == BootDevice::Cdrom {
                    eprintln!(
                        "guest requested reset: {kind:?} (disabling CD-first policy; booting HDD next)"
                    );
                    machine.set_boot_from_cd_if_present(false);
                    machine.set_boot_drive(0x80);
                    *cd_first_enabled = false;
                } else {
                    eprintln!("guest requested reset: {kind:?} (continuing)");
                }
                machine.reset();
                Ok(LoopControl::Continue)
            }
            RunExit::Assist { reason, .. } => {
                let ctx = exception_debug_context(machine);
                bail!("execution stopped: assist required: {reason:?} (total_executed={total_executed})\n  {ctx}")
            }
            RunExit::Exception { exception, .. } => {
                let ctx = exception_debug_context(machine);
                bail!("execution stopped: exception: {exception:?} (total_executed={total_executed})\n  {ctx}")
            }
            RunExit::CpuExit { exit, .. } => {
                let ctx = exception_debug_context(machine);
                bail!("execution stopped: cpu exit: {exit:?} (total_executed={total_executed})\n  {ctx}")
            }
        }
    }

    /// Dump one or more physical memory regions (`<addr>:<len>`, hex, comma-separated;
    /// each region capped at 0x400 bytes) in the `dump[...]` format used by the exception
    /// context. Shared by the fault path and the clean-exit probe below.
    fn dump_mem_regions(machine: &mut Machine, spec: &str) -> String {
        let mut out = String::new();
        for region in spec.split(',') {
            let mut parts = region.split(':');
            let addr = parts
                .next()
                .and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
            let len = parts
                .next()
                .and_then(|l| usize::from_str_radix(l.trim_start_matches("0x"), 16).ok());
            if let (Some(addr), Some(len)) = (addr, len) {
                let len = len.min(0x400);
                out.push_str(&format!(" | dump[{addr:#x}+{len:#x}]="));
                for i in 0..len {
                    let b = machine.read_physical_u8(addr.wrapping_add(i as u64));
                    if i > 0 && i % 16 == 0 {
                        out.push('|');
                    } else if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&format!("{b:02x}"));
                }
            }
        }
        out
    }

    /// Win32k session handle-table entry stride used by the HOOK type-5 walk
    /// (`cmp [rax+0xe], dl` / `add rax, 0x18` at `fffff9600001830x`).
    const TYPE5_ENTRY_STRIDE: u64 = 0x18;
    /// Byte offset of the object-type field inside a session handle-table entry.
    const TYPE5_TYPE_OFF: u64 = 0x0e;
    /// HOOK object type index in the win32k handle table.
    const TYPE5_HOOK_TYPE: u8 = 5;
    /// Historical table base (session space) — overridable via `AERO_TYPE5_TABLE_BASE`.
    const TYPE5_DEFAULT_TABLE_BASE: u64 = 0xffff_f900_c020_0000;
    /// Default number of entries to scan (covers historical count≈230 + margin).
    const TYPE5_DEFAULT_MAX_ENTRIES: u32 = 512;

    /// Pure decision for whether a type-5 (HOOK) handle-table entry should be
    /// freed to break the Find(0) restart livelock.
    ///
    /// `obj_raw` is the 8-byte object field (low 3 bits may be Ob attributes).
    /// `head_h` is `HEAD.h` from the pointed-to object, or `None` if unreadable.
    fn type5_hook_entry_is_bad(obj_raw: u64, head_h: Option<u64>) -> bool {
        let obj = obj_raw & !0x7;
        if obj == 0 {
            return true;
        }
        match head_h {
            None => true,
            Some(0) => true,
            Some(_) => false,
        }
    }

    /// Scan the win32k session handle table and free every type-5 (HOOK) entry that
    /// would livelock the Find walk:
    /// - object pointer is null (after attribute mask), or
    /// - object is present but `HEAD.h == 0` (null handle head → Find(0) restart).
    ///
    /// Also always re-clears historical entry 198 (VA `…c020129e`) for back-compat.
    /// Sticky: re-run every host slice via `AERO_AUTO_TYPE5_WEDGE`.
    ///
    /// Session space is often unmapped under System CR3 — we try the current CR3
    /// first, then a short list of known session/process DTBs from the live snap.
    fn wedge_type5_handle_table(machine: &mut Machine) -> Result<String> {
        let long_mode = (machine.cpu().msr.efer & (1 << 10)) != 0
            && (machine.cpu().control.cr0 & 0x8000_0000) != 0;
        if !long_mode {
            bail!("AERO_AUTO_TYPE5_WEDGE requires long mode with paging");
        }
        let table_base = std::env::var("AERO_TYPE5_TABLE_BASE")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(TYPE5_DEFAULT_TABLE_BASE);
        let max_entries = std::env::var("AERO_TYPE5_MAX")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(TYPE5_DEFAULT_MAX_ENTRIES)
            .min(4096);

        // Prefer a CR3 that actually maps session handle-table VA. System CR3 often
        // does not (session space is per-process). Try current, then recent hot DTBs
        // from the stdvga cold chain (csrss/winlogon/winpeshl).
        let mut candidates: Vec<u64> = vec![machine.cpu().control.cr3 & !0xfff];
        for &dtb in &[
            0x2b2f_b000u64, // winlogon
            0x2a60_1000u64, // winpeshl
            0x2c05_f000u64, // csrss session-0
            0x2b87_5000u64, // csrss session-1
            0x2b52_c000u64, // services
            0x187_000u64,   // System
        ] {
            if !candidates.contains(&dtb) {
                candidates.push(dtb);
            }
        }
        if let Ok(extra) = std::env::var("AERO_TYPE5_CR3") {
            if let Ok(c) = u64::from_str_radix(extra.trim().trim_start_matches("0x"), 16) {
                candidates.insert(0, c & !0xfff);
            }
        }

        let mut cr3 = candidates[0];
        let mut mapped = false;
        for &c in &candidates {
            if translate_long(machine, c, table_base).is_some() {
                cr3 = c;
                mapped = true;
                break;
            }
        }
        if !mapped {
            bail!("type5 table base {table_base:#x} unmapped under tried CR3s");
        }

        let mut cleared: Vec<u32> = Vec::new();
        let mut scanned_type5: u32 = 0;
        for idx in 0..max_entries {
            let entry = table_base.wrapping_add(u64::from(idx) * TYPE5_ENTRY_STRIDE);
            let typ_va = entry.wrapping_add(TYPE5_TYPE_OFF);
            let Some(typ_pa) = translate_long(machine, cr3, typ_va) else {
                break;
            };
            let typ = machine.read_physical_u8(typ_pa);
            if typ != TYPE5_HOOK_TYPE {
                continue;
            }
            scanned_type5 = scanned_type5.saturating_add(1);

            let mut obj_bytes = [0u8; 8];
            let mut obj_ok = true;
            for (j, b) in obj_bytes.iter_mut().enumerate() {
                let va = entry.wrapping_add(j as u64);
                match translate_long(machine, cr3, va) {
                    Some(pa) => *b = machine.read_physical_u8(pa),
                    None => {
                        obj_ok = false;
                        break;
                    }
                }
            }
            if !obj_ok {
                machine.write_physical_u8(typ_pa, 0);
                cleared.push(idx);
                continue;
            }
            let obj_raw = u64::from_le_bytes(obj_bytes);
            let obj = obj_raw & !0x7;

            let head_h = if obj == 0 {
                Some(0)
            } else {
                let mut head_bytes = [0u8; 8];
                let mut head_ok = true;
                for (j, b) in head_bytes.iter_mut().enumerate() {
                    let va = obj.wrapping_add(j as u64);
                    match translate_long(machine, cr3, va) {
                        Some(pa) => *b = machine.read_physical_u8(pa),
                        None => {
                            head_ok = false;
                            break;
                        }
                    }
                }
                if head_ok {
                    Some(u64::from_le_bytes(head_bytes))
                } else {
                    None
                }
            };

            if type5_hook_entry_is_bad(obj_raw, head_h) {
                machine.write_physical_u8(typ_pa, 0);
                cleared.push(idx);
            }
        }

        let legacy_va = 0xffff_f900_c020_129e_u64;
        if let Some(pa) = translate_long(machine, cr3, legacy_va) {
            if machine.read_physical_u8(pa) != 0 {
                machine.write_physical_u8(pa, 0);
                if !cleared.contains(&198) {
                    cleared.push(198);
                }
            }
        }

        if cleared.is_empty() {
            Ok(format!(
                " table={table_base:#x} cr3={cr3:#x} scanned_type5={scanned_type5} cleared=0"
            ))
        } else {
            let preview: Vec<String> = cleared.iter().take(16).map(|i| i.to_string()).collect();
            let more = if cleared.len() > 16 {
                format!(",…+{}", cleared.len() - 16)
            } else {
                String::new()
            };
            Ok(format!(
                " table={table_base:#x} cr3={cr3:#x} scanned_type5={scanned_type5} cleared={} idx=[{}{}]",
                cleared.len(),
                preview.join(","),
                more
            ))
        }
    }

    /// Write guest linear regions for bring-up experiments:
    /// `AERO_POKE_LINEAR=<va>:<hexbytes>[,...]` (hex VA, contiguous hex digits for bytes).
    /// Optional `AERO_POKE_CR3=<hex>` selects an alternate page-table root (process DTB),
    /// matching `AERO_DUMP_CR3` so user-mode VAs can be patched while the BSP sits in
    /// kernel CR3. Physical pokes: `AERO_POKE_PHYS=<pa>:<hexbytes>[,...]`.
    fn poke_linear_regions(machine: &mut Machine, spec: &str) -> Result<String> {
        let mut out = String::new();
        let long_mode = (machine.cpu().msr.efer & (1 << 10)) != 0
            && (machine.cpu().control.cr0 & 0x8000_0000) != 0;
        if !long_mode {
            bail!("AERO_POKE_LINEAR requires long mode with paging");
        }
        let cr3 = std::env::var("AERO_POKE_CR3")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| machine.cpu().control.cr3);
        if std::env::var_os("AERO_POKE_CR3").is_some() {
            out.push_str(&format!(" | cr3={cr3:#x}"));
        }
        for region in spec.split(',') {
            let mut parts = region.split(':');
            let addr = parts
                .next()
                .and_then(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok());
            let hex = parts.next().map(str::trim).unwrap_or("");
            let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
            if !hex.len().is_multiple_of(2) {
                bail!("AERO_POKE_LINEAR hexbyte length must be even: {hex}");
            }
            let Some(addr) = addr else {
                bail!("AERO_POKE_LINEAR bad region: {region}");
            };
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            for i in (0..hex.len()).step_by(2) {
                bytes.push(
                    u8::from_str_radix(&hex[i..i + 2], 16).with_context(|| {
                        format!("bad hex in AERO_POKE_LINEAR: {}", &hex[i..i + 2])
                    })?,
                );
            }
            for (i, b) in bytes.iter().enumerate() {
                let va = addr.wrapping_add(i as u64);
                let pa = translate_long(machine, cr3, va).ok_or_else(|| {
                    anyhow!("AERO_POKE_LINEAR unmapped va {va:#x} (cr3={cr3:#x})")
                })?;
                machine.write_physical_u8(pa, *b);
            }
            out.push_str(&format!(" | poke[{addr:#x}+{:#x}]=ok", bytes.len()));
        }
        Ok(out)
    }

    /// Physical-address poke: `AERO_POKE_PHYS=<pa>:<hexbytes>[,...]`.
    fn poke_phys_regions(machine: &mut Machine, spec: &str) -> Result<String> {
        let mut out = String::new();
        for region in spec.split(',') {
            let mut parts = region.split(':');
            let addr = parts
                .next()
                .and_then(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok());
            let hex = parts.next().map(str::trim).unwrap_or("");
            let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
            if !hex.len().is_multiple_of(2) {
                bail!("AERO_POKE_PHYS hexbyte length must be even: {hex}");
            }
            let Some(addr) = addr else {
                bail!("AERO_POKE_PHYS bad region: {region}");
            };
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            for i in (0..hex.len()).step_by(2) {
                bytes.push(
                    u8::from_str_radix(&hex[i..i + 2], 16).with_context(|| {
                        format!("bad hex in AERO_POKE_PHYS: {}", &hex[i..i + 2])
                    })?,
                );
            }
            for (i, b) in bytes.iter().enumerate() {
                machine.write_physical_u8(addr.wrapping_add(i as u64), *b);
            }
            out.push_str(&format!(" | poke_phys[{addr:#x}+{:#x}]=ok", bytes.len()));
        }
        Ok(out)
    }

    /// Parse hex u64 from `key=value` map entry.
    fn parse_hex_u64(s: &str) -> Result<u64> {
        let s = s.trim().trim_start_matches("0x").trim_start_matches("0X");
        u64::from_str_radix(s, 16).with_context(|| format!("bad hex: {s}"))
    }

    /// Force BSP into long-mode user (CPL3) at a chosen process context.
    /// Spec: `cr3=<hex>,rip=<hex>,rsp=<hex>[,rax=<hex>][,rcx=<hex>][,rdx=<hex>][,teb=<hex>]`
    fn force_user_resume(machine: &mut Machine, spec: &str) -> Result<String> {
        let mut cr3: Option<u64> = None;
        let mut rip: Option<u64> = None;
        let mut rsp: Option<u64> = None;
        let mut rax: u64 = 0;
        let mut rcx: Option<u64> = None;
        let mut rdx: Option<u64> = None;
        let mut rbx: Option<u64> = None;
        let mut teb: Option<u64> = None;
        let mut kernel = false;
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let mut kv = part.splitn(2, '=');
            let key = kv.next().unwrap_or("").trim().to_ascii_lowercase();
            let val = kv.next().unwrap_or("").trim();
            match key.as_str() {
                "cr3" => cr3 = Some(parse_hex_u64(val)?),
                "rip" => rip = Some(parse_hex_u64(val)?),
                "rsp" => rsp = Some(parse_hex_u64(val)?),
                "rax" => rax = parse_hex_u64(val)?,
                "rcx" => rcx = Some(parse_hex_u64(val)?),
                "rdx" => rdx = Some(parse_hex_u64(val)?),
                "rbx" => rbx = Some(parse_hex_u64(val)?),
                "teb" => teb = Some(parse_hex_u64(val)?),
                // kernel=1 → CPL0 (CS=0x10 / SS=0x18) so shellcode can store through
                // MmMapIoSpace LFB VAs in the high half.
                "kernel" => {
                    kernel = matches!(val, "1" | "true" | "yes" | "on" | "");
                }
                _ => bail!("AERO_FORCE_RESUME unknown key: {key}"),
            }
        }
        let cr3 = cr3.ok_or_else(|| anyhow!("AERO_FORCE_RESUME requires cr3="))?;
        let rip = rip.ok_or_else(|| anyhow!("AERO_FORCE_RESUME requires rip="))?;
        let rsp = rsp.ok_or_else(|| anyhow!("AERO_FORCE_RESUME requires rsp="))?;

        let core = machine.cpu_core_mut_by_index(0);
        let st = &mut core.state;

        // Preserve existing long-mode CR0/CR4/EFER; just switch CR3/CR8 and selectors.
        st.control.cr3 = cr3;
        st.control.cr8 = 0;
        st.control.cr2 = 0;

        const SEG_P: u32 = 1 << 7;
        const SEG_S: u32 = 1 << 4;
        const SEG_CODE: u32 = 0b1011; // code, execute/read, accessed
        const SEG_DATA: u32 = 0b0011; // data, r/w, accessed
        const SEG_L: u32 = 1 << 9;

        if kernel {
            // Win7 SP1 x64 kernel selectors: CS=0x10 (KGDT64_R0_CODE), SS=0x18.
            let dpl0 = 0u32 << 5;
            let cs_access = SEG_P | dpl0 | SEG_S | SEG_CODE | SEG_L;
            let ds_access = SEG_P | dpl0 | SEG_S | SEG_DATA;
            st.segments.cs.selector = 0x10;
            st.segments.cs.base = 0;
            st.segments.cs.limit = 0;
            st.segments.cs.access = cs_access;
            for seg in [
                &mut st.segments.ss,
                &mut st.segments.ds,
                &mut st.segments.es,
                &mut st.segments.fs,
            ] {
                seg.selector = 0x18;
                seg.base = 0;
                seg.limit = 0xfffff;
                seg.access = ds_access;
            }
            // Keep GS as PCR if already kernel; do not install TEB.
        } else {
            // Win7 SP1 x64 user selectors.
            // CS=0x33: GDT index 6, RPL=3, 64-bit code (L=1, DPL=3, present, code/read).
            // SS/DS/ES=0x2B: GDT index 5, RPL=3, data.
            let dpl3 = 3u32 << 5;
            let cs_access = SEG_P | dpl3 | SEG_S | SEG_CODE | SEG_L;
            let ds_access = SEG_P | dpl3 | SEG_S | SEG_DATA;

            st.segments.cs.selector = 0x33;
            st.segments.cs.base = 0;
            st.segments.cs.limit = 0;
            st.segments.cs.access = cs_access;

            for seg in [
                &mut st.segments.ss,
                &mut st.segments.ds,
                &mut st.segments.es,
            ] {
                seg.selector = 0x2b;
                seg.base = 0;
                seg.limit = 0xfffff;
                seg.access = ds_access;
            }

            // GS → TEB (user); KERNEL_GS_BASE keeps PCR so a later syscall swapgs works.
            if let Some(teb) = teb {
                let pcr = st.msr.gs_base; // current kernel GS (PCR) before we overwrite
                st.msr.kernel_gs_base = if pcr > 0xffff_8000_0000_0000 {
                    pcr
                } else {
                    st.msr.kernel_gs_base
                };
                st.msr.gs_base = teb;
                st.segments.gs.selector = 0x2b;
                st.segments.gs.base = teb;
                st.segments.gs.limit = 0xfffff;
                st.segments.gs.access = ds_access;
            }
        }

        // GPRs: RAX/RSP always; RCX/RDX only when given so a thread resume can
        // keep the wait handle sitting in the trapframe (csrss tid 628).
        st.gpr[0] = rax; // RAX
        st.gpr[4] = rsp; // RSP
        if let Some(v) = rcx {
            st.gpr[1] = v;
        } else if !kernel {
            st.gpr[1] = 0;
        }
        if let Some(v) = rdx {
            st.gpr[2] = v;
        } else if !kernel {
            st.gpr[2] = 0;
        }
        if let Some(v) = rbx {
            st.gpr[3] = v;
        } else if !kernel {
            st.gpr[3] = 0;
        }
        if !kernel {
            // Drop leftover kernel GPRs from the previous CurrentThread.
            for gpr in 8..12 {
                st.gpr[gpr] = 0;
            }
            st.gpr[6] = 0; // RSI
            st.gpr[7] = 0; // RDI
        }
        st.set_rip(rip);
        st.set_rflags(aero_cpu_core::state::RFLAGS_RESERVED1 | aero_cpu_core::state::RFLAGS_IF);
        st.halted = false;
        st.update_mode();

        Ok(format!(
            " cr3={cr3:#x} rip={rip:#x} rsp={rsp:#x} rax={rax:#x} rcx={rcx:?} rdx={rdx:?} rbx={rbx:?} teb={teb:?} cs={:#x} kernel={kernel} mode={:?}",
            st.segments.cs.selector,
            st.mode
        ))
    }

    /// Dump guest **linear** regions (`<va>:<len>`, hex, comma-separated; each capped
    /// at 0x100 bytes). Emits a compact page-table walk plus the bytes (unmapped `0xFE`).
    /// Optional `AERO_DUMP_CR3=<hex>` selects an alternate page-table root (process DTB).
    fn dump_linear_regions(machine: &mut Machine, spec: &str) -> String {
        let mut out = String::new();
        // Long mode: EFER.LMA (bit 10) + CR0.PG. Avoid depending on a re-exported CpuMode.
        let long_mode = (machine.cpu().msr.efer & (1 << 10)) != 0
            && (machine.cpu().control.cr0 & 0x8000_0000) != 0;
        let cr3 = std::env::var("AERO_DUMP_CR3")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| machine.cpu().control.cr3);
        if std::env::var_os("AERO_DUMP_CR3").is_some() {
            out.push_str(&format!(" | cr3={cr3:#x}"));
        }
        for region in spec.split(',') {
            let mut parts = region.split(':');
            let addr = parts
                .next()
                .and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
            let len = parts
                .next()
                .and_then(|l| usize::from_str_radix(l.trim_start_matches("0x"), 16).ok());
            if let (Some(addr), Some(len)) = (addr, len) {
                let len = len.min(0x100);
                out.push_str(&format!(" | va[{addr:#x}+{len:#x}]"));
                if long_mode {
                    out.push_str(&format!(" {}", walk_pt(machine, cr3, addr)));
                }
                out.push('=');
                let bytes = read_linear_bytes_cr3(machine, addr, len, long_mode, cr3);
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 && i % 16 == 0 {
                        out.push('|');
                    } else if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&format!("{b:02x}"));
                }
            }
        }
        out
    }

    /// Scan guest physical RAM for ASCII needles. Spec: comma-separated strings.
    /// Prints phys addrs of first hits (cap 12 per needle). Caps scan at 2 GiB.
    fn scan_ascii_in_ram(machine: &mut Machine, spec: &str) -> String {
        let needles: Vec<&[u8]> = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(8)
            .map(str::as_bytes)
            .filter(|b| !b.is_empty() && b.len() <= 32)
            .collect();
        if needles.is_empty() {
            return " (no needles)".into();
        }
        // CLI bring-up uses --ram; default scan 2 GiB (matches Win7 bring-up).
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut out = String::new();
        let mut hits: Vec<Vec<u64>> = vec![Vec::new(); needles.len()];
        let mut prev_tail = Vec::new();
        let mut addr: u64 = 0;
        while addr < ram_bytes {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            for (ni, needle) in needles.iter().enumerate() {
                if hits[ni].len() >= 12 {
                    continue;
                }
                let mut start = 0usize;
                while hits[ni].len() < 12 {
                    if let Some(rel) = find_bytes(&buf[start..], needle) {
                        let abs = base + (start + rel) as u64;
                        // Skip if this hit is entirely inside the prev_tail overlap region
                        // we already reported from the previous chunk.
                        if (abs >= addr || hits[ni].last().copied() != Some(abs))
                            && !hits[ni].contains(&abs)
                        {
                            hits[ni].push(abs);
                        }
                        start += rel + 1;
                    } else {
                        break;
                    }
                }
            }
            let keep = needles
                .iter()
                .map(|n| n.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
            prev_tail = if keep > 0 && buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }
        for (ni, needle) in needles.iter().enumerate() {
            let name = String::from_utf8_lossy(needle);
            out.push_str(&format!(" | '{name}':"));
            if hits[ni].is_empty() {
                out.push_str(" (none)");
            } else {
                for h in &hits[ni] {
                    out.push_str(&format!(" {h:#x}"));
                }
            }
        }
        out
    }

    fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// Scan guest physical RAM for little-endian integer needles (4 or 8 bytes).
    /// Spec: comma-separated hex values (optional `0x`). Reports first 16 hits each.
    fn scan_le_int_in_ram(machine: &mut Machine, spec: &str, width: usize) -> String {
        let width: usize = if width == 8 { 8 } else { 4 };
        let mut needles: Vec<(String, Vec<u8>)> = Vec::new();
        for part in spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(8)
        {
            let hex = part.trim_start_matches("0x").trim_start_matches("0X");
            let Ok(v) = u64::from_str_radix(hex, 16) else {
                continue;
            };
            let bytes = match width {
                8 => v.to_le_bytes().to_vec(),
                _ => (v as u32).to_le_bytes().to_vec(),
            };
            needles.push((format!("{v:#x}"), bytes));
        }
        if needles.is_empty() {
            return " (no needles)".into();
        }
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut hits: Vec<Vec<u64>> = vec![Vec::new(); needles.len()];
        let mut prev_tail = Vec::new();
        let mut addr: u64 = 0;
        while addr < ram_bytes {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            for (ni, (_name, needle)) in needles.iter().enumerate() {
                if hits[ni].len() >= 16 {
                    continue;
                }
                let mut start = 0usize;
                while hits[ni].len() < 16 {
                    if let Some(rel) = find_bytes(&buf[start..], needle) {
                        let abs = base + (start + rel) as u64;
                        if (abs >= addr || hits[ni].last().copied() != Some(abs))
                            && !hits[ni].contains(&abs)
                        {
                            hits[ni].push(abs);
                        }
                        start += rel + 1;
                    } else {
                        break;
                    }
                }
            }
            let keep = width.saturating_sub(1);
            prev_tail = if keep > 0 && buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }
        let mut out = String::new();
        for (ni, (name, _)) in needles.iter().enumerate() {
            out.push_str(&format!(" | '{name}':"));
            if hits[ni].is_empty() {
                out.push_str(" (none)");
            } else {
                for h in &hits[ni] {
                    out.push_str(&format!(" {h:#x}"));
                }
            }
        }
        out
    }

    /// Scan guest physical RAM for raw byte needles (hex strings, optional commas).
    fn scan_bytes_in_ram(machine: &mut Machine, spec: &str) -> String {
        let mut needles: Vec<(String, Vec<u8>)> = Vec::new();
        for part in spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(4)
        {
            let hex: String = part.chars().filter(|c| c.is_ascii_hexdigit()).collect();
            if hex.len() < 4 || !hex.len().is_multiple_of(2) || hex.len() > 64 {
                continue;
            }
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            let ok = (0..hex.len() / 2).all(|i| {
                u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                    .map(|b| bytes.push(b))
                    .is_ok()
            });
            if ok && !bytes.is_empty() {
                needles.push((hex, bytes));
            }
        }
        if needles.is_empty() {
            return " (no needles)".into();
        }
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut hits: Vec<Vec<u64>> = vec![Vec::new(); needles.len()];
        let mut prev_tail = Vec::new();
        let mut addr: u64 = 0;
        while addr < ram_bytes {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            for (ni, (_name, needle)) in needles.iter().enumerate() {
                if hits[ni].len() >= 16 {
                    continue;
                }
                let mut start = 0usize;
                while hits[ni].len() < 16 {
                    if let Some(rel) = find_bytes(&buf[start..], needle) {
                        let abs = base + (start + rel) as u64;
                        if (abs >= addr || hits[ni].last().copied() != Some(abs))
                            && !hits[ni].contains(&abs)
                        {
                            hits[ni].push(abs);
                        }
                        start += rel + 1;
                    } else {
                        break;
                    }
                }
            }
            let keep = needles
                .iter()
                .map(|(_, n)| n.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
            prev_tail = if keep > 0 && buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }
        let mut out = String::new();
        for (ni, (name, _)) in needles.iter().enumerate() {
            out.push_str(&format!(" | '{name}':"));
            if hits[ni].is_empty() {
                out.push_str(" (none)");
            } else {
                for h in &hits[ni] {
                    out.push_str(&format!(" {h:#x}"));
                }
            }
        }
        out
    }

    /// Heuristic Win7 SURFOBJ scan: sizlBitmap=(1024,768) at +0x20.
    /// Includes device surfaces (cjBits may be 0) and bitmaps (cjBits=0x300000).
    /// Reports phys of SURFOBJ base, pvBits, pvScan0, lDelta, iBitmapFormat, iType.
    fn scan_surfobj_candidates(machine: &mut Machine) -> String {
        // Pattern at SURFOBJ+0x20: cx=0x400, cy=0x300 (device surfaces may have cjBits=0)
        let needle: [u8; 8] = [
            0x00, 0x04, 0x00, 0x00, // 1024
            0x00, 0x03, 0x00, 0x00, // 768
        ];
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut hits = Vec::new();
        let mut prev_tail = Vec::new();
        let mut addr: u64 = 0;
        while addr < ram_bytes && hits.len() < 32 {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            let mut start = 0usize;
            while hits.len() < 32 {
                if let Some(rel) = find_bytes(&buf[start..], &needle) {
                    let abs = base + (start + rel) as u64;
                    // SURFOBJ base is 0x20 before sizlBitmap
                    let surf = abs.saturating_sub(0x20);
                    // Filter: plausible format/type at +0x48/+0x4c (fmt 0..8, type 0..3)
                    let blob = machine.read_physical_bytes(surf, 0x50);
                    if blob.len() >= 0x50 {
                        let i_fmt =
                            u32::from_le_bytes(blob[0x48..0x4c].try_into().unwrap_or([0; 4]));
                        let i_type =
                            u16::from_le_bytes(blob[0x4c..0x4e].try_into().unwrap_or([0; 2]));
                        let cj = u32::from_le_bytes(blob[0x28..0x2c].try_into().unwrap_or([0; 4]));
                        // STYPE_DEVICE=0, BITMAP=1, DEVBITMAP=3; BMF formats 1..8
                        if i_type <= 3
                            && i_fmt <= 8
                            && (cj == 0 || cj == 0x30_0000 || cj == 0x18_0000)
                            && !hits.contains(&surf)
                        {
                            hits.push(surf);
                        }
                    }
                    start += rel + 1;
                } else {
                    break;
                }
            }
            let keep = needle.len().saturating_sub(1);
            prev_tail = if keep > 0 && buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }
        if hits.is_empty() {
            return " (none)".into();
        }
        let mut out = String::new();
        for surf in hits {
            let blob = machine.read_physical_bytes(surf, 0x50);
            if blob.len() < 0x50 {
                continue;
            }
            let cj = u32::from_le_bytes(blob[0x28..0x2c].try_into().unwrap_or([0; 4]));
            let pv_bits = u64::from_le_bytes(blob[0x30..0x38].try_into().unwrap_or([0; 8]));
            let pv_scan0 = u64::from_le_bytes(blob[0x38..0x40].try_into().unwrap_or([0; 8]));
            let l_delta = i32::from_le_bytes(blob[0x40..0x44].try_into().unwrap_or([0; 4]));
            let i_fmt = u32::from_le_bytes(blob[0x48..0x4c].try_into().unwrap_or([0; 4]));
            let i_type = u16::from_le_bytes(blob[0x4c..0x4e].try_into().unwrap_or([0; 2]));
            out.push_str(&format!(
                " | surf_pa={surf:#x} cj={cj:#x} pvBits={pv_bits:#x} pvScan0={pv_scan0:#x} lDelta={l_delta} fmt={i_fmt:#x} type={i_type:#x}"
            ));
        }
        out
    }

    /// Scan guest physical RAM for UTF-16LE encodings of the given strings.
    /// Spec: comma-separated (max 8, each ≤32 UTF-16 code units).
    fn scan_utf16_in_ram(machine: &mut Machine, spec: &str) -> String {
        let needles: Vec<(String, Vec<u8>)> = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(8)
            .filter(|s| s.chars().count() <= 32)
            .map(|s| {
                let mut enc = Vec::with_capacity(s.len() * 2);
                for u in s.encode_utf16() {
                    enc.extend_from_slice(&u.to_le_bytes());
                }
                (s.to_string(), enc)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect();
        if needles.is_empty() {
            return " (no needles)".into();
        }
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut out = String::new();
        let mut hits: Vec<Vec<u64>> = vec![Vec::new(); needles.len()];
        let mut prev_tail = Vec::new();
        let mut addr: u64 = 0;
        while addr < ram_bytes {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            for (ni, (_name, needle)) in needles.iter().enumerate() {
                if hits[ni].len() >= 12 {
                    continue;
                }
                let mut start = 0usize;
                while hits[ni].len() < 12 {
                    if let Some(rel) = find_bytes(&buf[start..], needle) {
                        let abs = base + (start + rel) as u64;
                        if (abs >= addr || hits[ni].last().copied() != Some(abs))
                            && !hits[ni].contains(&abs)
                        {
                            hits[ni].push(abs);
                        }
                        start += rel + 2; // step by UTF-16 unit
                    } else {
                        break;
                    }
                }
            }
            let keep = needles
                .iter()
                .map(|(_, n)| n.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
            prev_tail = if keep > 0 && buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }
        for (ni, (name, _)) in needles.iter().enumerate() {
            out.push_str(&format!(" | '{name}':"));
            if hits[ni].is_empty() {
                out.push_str(" (none)");
            } else {
                for h in &hits[ni] {
                    out.push_str(&format!(" {h:#x}"));
                }
            }
        }
        out
    }

    /// Walk guest page tables looking for VAs that map `phys` (AERO_FIND_PHYS=<hex>).
    /// Searches System DTB `0x187000`, the current CR3, and optional comma-separated
    /// `AERO_FIND_PHYS_CR3` values. Reports up to 32 hits.
    fn find_phys_maps(machine: &mut Machine, spec: &str) -> Result<String> {
        let phys = u64::from_str_radix(spec.trim().trim_start_matches("0x"), 16)
            .with_context(|| format!("bad AERO_FIND_PHYS address: {spec}"))?;
        let target_2m = phys & !0x1f_ffff;
        let target_4k = phys & !0xfff;
        let mut cr3s: Vec<(u64, &'static str)> = vec![
            (0x187000, "System"),
            (machine.cpu().control.cr3 & !0xfff, "current"),
        ];
        if let Ok(extra) = std::env::var("AERO_FIND_PHYS_CR3") {
            for raw in extra.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let cr3 = u64::from_str_radix(raw.trim_start_matches("0x"), 16)
                    .with_context(|| format!("bad AERO_FIND_PHYS_CR3 address: {raw}"))?;
                cr3s.push((cr3 & !0xfff, "extra"));
            }
        }
        // Dedup
        cr3s.dedup_by_key(|(c, _)| *c);

        let mut out = String::new();
        let mut hits = 0usize;
        const MAX_HITS: usize = 32;

        for (cr3, tag) in cr3s {
            let pml4 = cr3 & 0x000f_ffff_ffff_f000;
            // Low half (identity/user) + high half (kernel). Skip unused middle.
            let pml4_indices: Vec<u64> = (0u64..16).chain(256u64..512).collect();
            for pml4_i in pml4_indices {
                let pml4e = machine.read_physical_u64(pml4 + pml4_i * 8);
                if pml4e & 1 == 0 {
                    continue;
                }
                let pdpt = pml4e & 0x000f_ffff_ffff_f000;
                for pdpt_i in 0u64..512 {
                    let pdpte = machine.read_physical_u64(pdpt + pdpt_i * 8);
                    if pdpte & 1 == 0 {
                        continue;
                    }
                    // 1GiB page
                    if pdpte & 0x80 != 0 {
                        let base = pdpte & 0x000f_ffff_c000_0000;
                        if phys >= base && phys < base + 0x4000_0000 {
                            let va = (pml4_i << 39) | (pdpt_i << 30) | (phys & 0x3fff_ffff);
                            // Sign-extend canonical VA
                            let va = if va & (1u64 << 47) != 0 {
                                va | 0xffff_0000_0000_0000
                            } else {
                                va
                            };
                            out.push_str(&format!(
                                " | {tag} va={va:#x} -> {phys:#x} (1G) pte={pdpte:#x}"
                            ));
                            hits += 1;
                            if hits >= MAX_HITS {
                                out.push_str(&format!(" | (capped at {MAX_HITS})"));
                                return Ok(out);
                            }
                        }
                        continue;
                    }
                    let pd = pdpte & 0x000f_ffff_ffff_f000;
                    for pd_i in 0u64..512 {
                        let pde = machine.read_physical_u64(pd + pd_i * 8);
                        if pde & 1 == 0 {
                            continue;
                        }
                        if pde & 0x80 != 0 {
                            // 2MiB
                            let base = pde & 0x000f_ffff_ffe0_0000;
                            if base == target_2m {
                                let va = (pml4_i << 39)
                                    | (pdpt_i << 30)
                                    | (pd_i << 21)
                                    | (phys & 0x1f_ffff);
                                let va = if va & (1u64 << 47) != 0 {
                                    va | 0xffff_0000_0000_0000
                                } else {
                                    va
                                };
                                out.push_str(&format!(
                                    " | {tag} va={va:#x} -> {phys:#x} (2M) pte={pde:#x}"
                                ));
                                hits += 1;
                                if hits >= MAX_HITS {
                                    out.push_str(&format!(" | (capped at {MAX_HITS})"));
                                    return Ok(out);
                                }
                            }
                            continue;
                        }
                        let pt = pde & 0x000f_ffff_ffff_f000;
                        for pt_i in 0u64..512 {
                            let pte = machine.read_physical_u64(pt + pt_i * 8);
                            if pte & 1 == 0 {
                                continue;
                            }
                            let base = pte & 0x000f_ffff_ffff_f000;
                            if base == target_4k {
                                let va = (pml4_i << 39)
                                    | (pdpt_i << 30)
                                    | (pd_i << 21)
                                    | (pt_i << 12)
                                    | (phys & 0xfff);
                                let va = if va & (1u64 << 47) != 0 {
                                    va | 0xffff_0000_0000_0000
                                } else {
                                    va
                                };
                                out.push_str(&format!(
                                    " | {tag} va={va:#x} -> {phys:#x} (4K) pte={pte:#x}"
                                ));
                                hits += 1;
                                if hits >= MAX_HITS {
                                    out.push_str(&format!(" | (capped at {MAX_HITS})"));
                                    return Ok(out);
                                }
                            }
                        }
                    }
                }
            }
        }
        if hits == 0 {
            out.push_str(&format!(
                " (no VA maps phys {phys:#x} in System/current CR3)"
            ));
        } else {
            out.push_str(&format!(" | hits={hits}"));
        }
        Ok(out)
    }

    /// Win7 bootvid MmMapIoSpace of the VBE LFB (observed live on STDVGA cold chain).
    /// Prefer redirecting Eng here over forging system-PTE MMIO maps (`AERO_MAP_LFB_VA` → 0x1A).
    const BOOTVID_LFB_KVA: u64 = 0xffff_f7ff_efd0_0000;

    /// Rewrite Eng FrameBufferBase + 1024×768 SURFOBJ.pvBits to the live bootvid KVA.
    ///
    /// Diagnosis (bar0-setup): Eng structure at phys has FrameBufferBase
    /// `0xfffff88003c00000` with PDE NP; bootvid KVA still maps LFB. SURFOBJ has
    /// null pvBits. Pointing both at the Windows-owned bootvid map restores a
    /// present path without forged MMIO PTEs.
    ///
    /// Returns `(message, eng_struct_pa, surfobj_pa)`. When `cached` is Some,
    /// only re-pokes those addresses (no full RAM scan).
    fn redir_eng_lfb_to_bootvid_kva(
        machine: &mut Machine,
        cached: Option<(Option<u64>, Option<u64>)>,
    ) -> Result<(String, Option<u64>, Option<u64>)> {
        let lfb = machine.vbe_lfb_base();
        if lfb == 0 {
            bail!("VBE LFB base is 0");
        }

        let poke_eng = |m: &mut Machine, pa: u64| -> bool {
            let old = m.read_physical_u64(pa + 0x18);
            if old != BOOTVID_LFB_KVA {
                m.write_physical_u64(pa + 0x18, BOOTVID_LFB_KVA);
                true
            } else {
                false
            }
        };
        // When set, promote the screen-sized STYPE_BITMAP to STYPE_DEVICE and
        // clear cjBits — framebuf DrvEnableSurface / EngModifySurface failed to
        // leave a real device primary (only a null-bit bitmap remains).
        let force_device = std::env::var_os("AERO_FORCE_DEVICE_SURF").is_some();
        let poke_surf = |m: &mut Machine, pa: u64| -> bool {
            let mut changed = false;
            let pv = m.read_physical_u64(pa + 0x30);
            if pv != BOOTVID_LFB_KVA {
                m.write_physical_u64(pa + 0x30, BOOTVID_LFB_KVA);
                m.write_physical_u64(pa + 0x38, BOOTVID_LFB_KVA);
                let cur = m.read_physical_u64(pa + 0x40);
                m.write_physical_u64(pa + 0x40, (cur & !0xffff_ffff) | 0x1000);
                changed = true;
            }
            if force_device {
                // iType at +0x4c (UShort): STYPE_DEVICE=0 (was STYPE_BITMAP=1).
                let t = m.read_physical_u64(pa + 0x48);
                // low 32 = iBitmapFormat, next 16 = iType, next 16 = fjBitmap
                let fmt = t & 0xffff_ffff;
                let i_type = (t >> 32) & 0xffff;
                let fj = (t >> 48) & 0xffff;
                if i_type != 0 {
                    // Keep BMF_32BPP (6); force STYPE_DEVICE; fjBitmap=0.
                    let new = fmt | (fj << 48);
                    m.write_physical_u64(pa + 0x48, new);
                    // cjBits=0 is typical for device surfaces.
                    let mid = m.read_physical_u64(pa + 0x28);
                    m.write_physical_u64(pa + 0x28, mid & !0xffff_ffff);
                    changed = true;
                }
            }
            changed
        };

        if let Some((eng_pa, surf_pa)) = cached {
            let mut eng_hits = 0u32;
            let mut surf_hits = 0u32;
            if let Some(pa) = eng_pa {
                if poke_eng(machine, pa) {
                    eng_hits = 1;
                }
            }
            if let Some(pa) = surf_pa {
                if poke_surf(machine, pa) {
                    surf_hits = 1;
                }
            }
            return Ok((
                format!(
                    " (sticky) bootvid_kva={:#x} eng_pa={:#x} eng_rewrites={eng_hits} surf_pa={:#x} surf_rewrites={surf_hits}",
                    BOOTVID_LFB_KVA,
                    eng_pa.unwrap_or(0),
                    surf_pa.unwrap_or(0),
                ),
                eng_pa,
                surf_pa,
            ));
        }

        // Also accept BAR0 (STDVGA) and classic BAR1+LFB phys encodings.
        let bar0 = machine.aerogpu_bar0_base().unwrap_or(0);
        let mut phys_needles = vec![lfb];
        if bar0 != 0 && bar0 != lfb {
            phys_needles.push(bar0);
        }
        // Warm snaps may still have Eng phys = e4040000 while live VBE base moved.
        if !phys_needles.contains(&0xe404_0000) {
            phys_needles.push(0xe404_0000);
        }

        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);

        let mut eng_hits = 0u32;
        let mut surf_hits = 0u32;
        let mut eng_pa: Option<u64> = None;
        let mut surf_pa: Option<u64> = None;

        // Scan for Eng mode/fb descriptor: u64 phys + u32 1024 + u32 768
        // at [0]=phys, [8]=w, [0xc]=h, [0x18]=FrameBufferBase VA.
        let chunk = 1024 * 1024usize;
        let mut addr = 0u64;
        let mut prev_tail = Vec::new();
        while addr < ram_bytes {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let mut buf = machine.read_physical_bytes(addr, len);
            if !prev_tail.is_empty() {
                let mut combined = prev_tail.clone();
                combined.extend_from_slice(&buf);
                buf = combined;
            }
            let base = addr.saturating_sub(prev_tail.len() as u64);
            let mut i = 0usize;
            while i + 0x20 <= buf.len() {
                let pa_here = base + i as u64;
                // Eng descriptor: phys needle + 1024x768
                if i + 0x20 <= buf.len() {
                    let phys = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
                    if phys_needles.contains(&phys) {
                        let w = u32::from_le_bytes(buf[i + 8..i + 12].try_into().unwrap());
                        let h = u32::from_le_bytes(buf[i + 12..i + 16].try_into().unwrap());
                        if w == 1024 && h == 768 {
                            if poke_eng(machine, pa_here) {
                                eng_hits += 1;
                            }
                            eng_pa.get_or_insert(pa_here);
                        }
                    }
                }
                // SURFOBJ: sizlBitmap 1024x768 at +0x20, cjBits 0x300000, BMF_32BPP
                if i + 0x50 <= buf.len() {
                    let cx = u32::from_le_bytes(buf[i + 0x20..i + 0x24].try_into().unwrap());
                    let cy = u32::from_le_bytes(buf[i + 0x24..i + 0x28].try_into().unwrap());
                    let cj = u32::from_le_bytes(buf[i + 0x28..i + 0x2c].try_into().unwrap());
                    let i_fmt = u32::from_le_bytes(buf[i + 0x48..i + 0x4c].try_into().unwrap());
                    let i_type = u16::from_le_bytes(buf[i + 0x4c..i + 0x4e].try_into().unwrap());
                    if cx == 1024 && cy == 768 && cj == 0x30_0000 && i_fmt == 6 && i_type == 1 {
                        if poke_surf(machine, pa_here) {
                            surf_hits += 1;
                        }
                        surf_pa.get_or_insert(pa_here);
                    }
                }
                i += 8;
            }
            let keep = 0x50usize;
            prev_tail = if buf.len() >= keep {
                buf[buf.len() - keep..].to_vec()
            } else {
                Vec::new()
            };
            addr += len as u64;
        }

        if eng_pa.is_none() && surf_pa.is_none() {
            bail!("no Eng FrameBuffer descriptor or 1024x768 SURFOBJ found");
        }
        Ok((
            format!(
                " bootvid_kva={:#x} eng_pa={:#x} eng_rewrites={eng_hits} surf_pa={:#x} surf_rewrites={surf_hits} lfb_phys={lfb:#x}",
                BOOTVID_LFB_KVA,
                eng_pa.unwrap_or(0),
                surf_pa.unwrap_or(0),
            ),
            eng_pa,
            surf_pa,
        ))
    }

    /// Identity-map the VBE LFB GPA into the current CR3 as one 2 MiB large page.
    ///
    /// Used for paint-wall diagnosis: after boot the guest leaves the LFB GPA unmapped
    /// (user stores `#PF` with P=0). Installing a RW+PCD+PWT+US large page lets shellcode
    /// and (eventually) GDI hit AeroGPU BAR1 / VGA LFB MMIO so `AERO_COUNT_LFB_WRITES` moves.
    fn map_vbe_lfb_identity_2m(machine: &mut Machine) -> Result<String> {
        let lfb = machine.vbe_lfb_base();
        if lfb == 0 {
            bail!("VBE LFB base is 0");
        }
        // 2 MiB region covering the LFB.
        let phys_2m = lfb & !0x1f_ffff;
        let cr3 = machine.cpu().control.cr3 & !0xfff;
        let pml4_i = (phys_2m >> 39) & 0x1ff;
        let pdpt_i = (phys_2m >> 30) & 0x1ff;
        let pd_i = (phys_2m >> 21) & 0x1ff;

        let read_pte = |m: &mut Machine, table: u64, idx: u64| -> u64 {
            m.read_physical_u64(table.wrapping_add(idx * 8))
        };
        let write_pte = |m: &mut Machine, table: u64, idx: u64, val: u64| {
            m.write_physical_u64(table.wrapping_add(idx * 8), val);
        };

        // Locate a free 4 KiB physical page for a new table (scan high RAM for zeros).
        let find_free_page = |m: &mut Machine| -> Result<u64> {
            // Prefer 16–32 MiB region which is usually free of firmware tables on our 2 GiB guests.
            for page in (0x10_0000u64..0x40_0000).step_by(0x1000) {
                let mut zero = true;
                for off in (0..0x1000).step_by(8) {
                    if m.read_physical_u64(page + off as u64) != 0 {
                        zero = false;
                        break;
                    }
                }
                if zero {
                    return Ok(page);
                }
            }
            // Fall back: carve from top of low 64 MiB.
            let mut page = 0x80_0000u64;
            while page > 0x20_0000 {
                page -= 0x1000;
                let mut zero = true;
                for off in (0..0x1000).step_by(8) {
                    if m.read_physical_u64(page + off as u64) != 0 {
                        zero = false;
                        break;
                    }
                }
                if zero {
                    return Ok(page);
                }
            }
            bail!("no free zero page for page-table allocation");
        };

        let pml4e = read_pte(machine, cr3, pml4_i);
        if pml4e & 1 == 0 {
            bail!("PML4[{pml4_i}] not present (cr3={cr3:#x})");
        }
        let pdpt = pml4e & 0x000f_ffff_ffff_f000;

        let mut pdpte = read_pte(machine, pdpt, pdpt_i);
        let pd = if pdpte & 1 == 0 {
            // Allocate a fresh PD.
            let pd_pa = find_free_page(machine)?;
            // Present + Writable + User (so CPL3 paint tests work) + Accessed.
            write_pte(machine, pdpt, pdpt_i, pd_pa | 0x7);
            pdpte = pd_pa | 0x7;
            pd_pa
        } else if pdpte & (1 << 7) != 0 {
            bail!("PDPT[{pdpt_i}] is a 1 GiB page; refusing to split");
        } else {
            pdpte & 0x000f_ffff_ffff_f000
        };

        // PDE: 2 MiB large page. Flags: P|RW|US|PWT|PCD|A|D|PS
        // 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 = 0xFF
        let pde_flags = 0xFFu64;
        let pde = phys_2m | pde_flags;
        let old = read_pte(machine, pd, pd_i);
        write_pte(machine, pd, pd_i, pde);

        // Flush TLB by rewriting CR3.
        let cr3_now = machine.cpu().control.cr3;
        machine.cpu_mut().control.cr3 = cr3_now;

        Ok(format!(
            " cr3={cr3:#x} lfb={lfb:#x} map_2m={phys_2m:#x} PML4[{pml4_i}]=present PDPT[{pdpt_i}]={pdpte:#x} PDE[{pd_i}] {old:#x}->{pde:#x}"
        ))
    }

    /// Map `va` → VBE LFB phys with 4 KiB PTEs for a full 1024×768×32 FB (~3 MiB).
    ///
    /// Bring-up: Eng/miniport may retain FrameBufferBase in system-PTE space while the
    /// PDE is NP (session thrash). LFB phys is not 2 MiB-aligned (`…4040000`), so this
    /// uses 4 KiB pages: `va+0` → `lfb+0`. Applies under **System** DTB `0x187000` and
    /// the *current* CR3 (shared kernel half is not always shared for new PT pages).
    fn map_va_to_vbe_lfb_2m(machine: &mut Machine, va: u64) -> Result<String> {
        let lfb = machine.vbe_lfb_base();
        if lfb == 0 {
            bail!("VBE LFB base is 0");
        }
        const FB_BYTES: u64 = 0x30_0000;
        let va_base = va & !0xfff;
        let phys_base = lfb & !0xfff;
        let cur = machine.cpu().control.cr3 & !0xfff;
        let mut cr3s = vec![0x187000u64, cur];
        cr3s.dedup();

        let read_pte = |m: &mut Machine, table: u64, idx: u64| -> u64 {
            m.read_physical_u64(table.wrapping_add(idx * 8))
        };
        let write_pte = |m: &mut Machine, table: u64, idx: u64, val: u64| {
            m.write_physical_u64(table.wrapping_add(idx * 8), val);
        };

        let find_free_page = |m: &mut Machine| -> Result<u64> {
            for page in (0x10_0000u64..0x40_0000).step_by(0x1000) {
                let mut zero = true;
                for off in (0..0x1000).step_by(8) {
                    if m.read_physical_u64(page + off as u64) != 0 {
                        zero = false;
                        break;
                    }
                }
                if zero {
                    return Ok(page);
                }
            }
            let mut page = 0x80_0000u64;
            while page > 0x20_0000 {
                page -= 0x1000;
                let mut zero = true;
                for off in (0..0x1000).step_by(8) {
                    if m.read_physical_u64(page + off as u64) != 0 {
                        zero = false;
                        break;
                    }
                }
                if zero {
                    return Ok(page);
                }
            }
            bail!("no free zero page for page-table allocation");
        };

        let ensure_pt = |m: &mut Machine, cr3: u64, page_va: u64| -> Result<u64> {
            let pml4_i = (page_va >> 39) & 0x1ff;
            let pdpt_i = (page_va >> 30) & 0x1ff;
            let pd_i = (page_va >> 21) & 0x1ff;

            let pml4e = read_pte(m, cr3, pml4_i);
            let pdpt = if pml4e & 1 == 0 {
                let pa = find_free_page(m)?;
                write_pte(m, cr3, pml4_i, pa | 0x7);
                pa
            } else {
                pml4e & 0x000f_ffff_ffff_f000
            };

            let pdpte = read_pte(m, pdpt, pdpt_i);
            let pd = if pdpte & 1 == 0 {
                let pa = find_free_page(m)?;
                write_pte(m, pdpt, pdpt_i, pa | 0x7);
                pa
            } else if pdpte & (1 << 7) != 0 {
                bail!("PDPT[{pdpt_i}] is a 1 GiB page; refusing to split");
            } else {
                pdpte & 0x000f_ffff_ffff_f000
            };

            let pde = read_pte(m, pd, pd_i);
            let pt = if pde & 1 == 0 {
                let pa = find_free_page(m)?;
                write_pte(m, pd, pd_i, pa | 0x27);
                pa
            } else if pde & (1 << 7) != 0 {
                bail!("PDE[{pd_i}] is a 2 MiB page; refusing to split at {page_va:#x}");
            } else {
                pde & 0x000f_ffff_ffff_f000
            };
            Ok(pt)
        };

        let pte_flags = 0x7Fu64;
        let pages = (FB_BYTES / 0x1000) as usize;
        let mut parts = Vec::new();
        for cr3 in cr3s {
            let mut mapped = 0usize;
            for i in 0..pages {
                let page_va = va_base + (i as u64) * 0x1000;
                let page_pa = phys_base + (i as u64) * 0x1000;
                let pt = ensure_pt(machine, cr3, page_va)?;
                let pt_i = (page_va >> 12) & 0x1ff;
                write_pte(machine, pt, pt_i, page_pa | pte_flags);
                mapped += 1;
            }
            parts.push(format!("cr3={cr3:#x}:pages={mapped}"));
        }

        let cr3_now = machine.cpu().control.cr3;
        machine.cpu_mut().control.cr3 = cr3_now;

        Ok(format!(
            " va={va:#x} lfb={lfb:#x} bytes={FB_BYTES:#x} (4K) {}",
            parts.join(" ")
        ))
    }

    /// Redirect winpeshl's WpeUtil IAT slots to `xor eax,eax; ret` so SCE RPC
    /// (`WpeInstallServicesSecurityTemplate` etc.) never blocks on scerpc.
    ///
    /// Fixed PE RVAs (Win7 SP1 winpeshl.exe): WpeUtil IAT at `ImageBase+0x1240`
    /// (6 slots). Gadget written at `ImageBase+0x7f00` (inside image, RW from
    /// prior inject experiments). Sticky: re-apply each host slice.
    ///
    /// Caches (dtb, base) after first success so we do not re-scan 2 GiB RAM
    /// every slice (that drops throughput ~100×).
    fn wedge_sce_wpeutil_iat(machine: &mut Machine) -> Result<String> {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::OnceLock;
        static CACHED: OnceLock<(u64, u64, u64, u64)> = OnceLock::new(); // (pid, dtb, base, gadget)
        static MISS_TICKS: AtomicU64 = AtomicU64::new(0);
        static LOGGED_OK: AtomicBool = AtomicBool::new(false);

        let long_mode = (machine.cpu().msr.efer & (1 << 10)) != 0
            && (machine.cpu().control.cr0 & 0x8000_0000) != 0;
        if !long_mode {
            bail!("AERO_AUTO_SCE_WEDGE requires long mode with paging");
        }

        /// Resolve winpeshl ImageBase under `dtb` (PEB candidates + MZ/PE scan).
        fn resolve_image_base(machine: &mut Machine, dtb: u64) -> Option<u64> {
            for peb_cand in [
                0x07ff_fffd_b000_u64,
                0x07ff_fffd_a000_u64,
                0x07ff_fffd_c000_u64,
                0x07ff_fffd_9000_u64,
                0x07ff_fffd_8000_u64,
                0x07ff_fffd_d000_u64,
                0x07ff_fffd_e000_u64 - 0x3000,
            ] {
                let mut out = [0u8; 0x18];
                let mut ok = true;
                for (i, byte) in out.iter_mut().enumerate() {
                    match translate_long(machine, dtb, peb_cand + i as u64) {
                        Some(pa) => *byte = machine.read_physical_bytes(pa, 1)[0],
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                let ib = u64::from_le_bytes(out[0x10..0x18].try_into().unwrap());
                if (0xff00_0000..0xfff0_0000).contains(&ib) {
                    if let Some(pa) = translate_long(machine, dtb, ib) {
                        let mz = machine.read_physical_bytes(pa, 2);
                        if mz == [b'M', b'Z'] {
                            return Some(ib);
                        }
                    }
                }
            }
            // Dense scan of typical ASLR high user half for winpeshl PE.
            let mut va = 0xff00_0000u64;
            while va < 0xfff0_0000 {
                if let Some(pa) = translate_long(machine, dtb, va) {
                    let mz = machine.read_physical_bytes(pa, 2);
                    if mz == [b'M', b'Z'] {
                        let e_lf = machine.read_physical_bytes(pa + 0x3c, 4);
                        let e_lfanew = u32::from_le_bytes(e_lf.try_into().unwrap_or([0; 4])) as u64;
                        if e_lfanew < 0x400 {
                            if let Some(pe_pa) = translate_long(machine, dtb, va + e_lfanew) {
                                let pe = machine.read_physical_bytes(pe_pa, 4);
                                if pe == *b"PE\0\0" {
                                    if let Some(opt_pa) =
                                        translate_long(machine, dtb, va + e_lfanew + 24 + 0x38)
                                    {
                                        let soi = u32::from_le_bytes(
                                            machine
                                                .read_physical_bytes(opt_pa, 4)
                                                .try_into()
                                                .unwrap_or([0; 4]),
                                        );
                                        if (0x70_000..0x120_000).contains(&soi) {
                                            return Some(va);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                va += 0x1_0000;
            }
            None
        }

        /// True if `base` under `dtb` looks like winpeshl (WpeUtil IAT mapped + sane).
        fn wpeutil_iat_sane(machine: &mut Machine, dtb: u64, base: u64) -> bool {
            let Some(iat0) = translate_long(machine, dtb, base + 0x1240) else {
                return false;
            };
            let slot0 = u64::from_le_bytes(
                machine
                    .read_physical_bytes(iat0, 8)
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            // Reject empty / low pointers (not a real user import).
            //
            // This was written as a conjunction with `< 0x7f00_0000_0000`, which the low bound
            // already implies, so only the low bound ever decided anything. If an upper bound on
            // user-space addresses was intended, it was never in effect.
            if slot0 < 0x7000_0000 {
                return false;
            }
            // Require at least 3 non-zero IAT slots (WpeUtil has 6).
            let mut nz = 0u32;
            for slot in 0..6u64 {
                let Some(pa) = translate_long(machine, dtb, base + 0x1240 + slot * 8) else {
                    break;
                };
                let v = u64::from_le_bytes(
                    machine
                        .read_physical_bytes(pa, 8)
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                if v == 0 {
                    break;
                }
                nz += 1;
            }
            nz >= 3
        }

        /// Read PEB→ProcessParameters→ImagePathName and require "winpeshl" (UTF-16).
        /// Prevents patching services.exe / other high-user PEs that share IAT-shaped holes.
        fn peb_path_is_winpeshl(machine: &mut Machine, dtb: u64) -> bool {
            const PEB_CANDS: [u64; 7] = [
                0x07ff_fffd_b000,
                0x07ff_fffd_a000,
                0x07ff_fffd_c000,
                0x07ff_fffd_9000,
                0x07ff_fffd_8000,
                0x07ff_fffd_d000,
                0x07ff_fffd_e000 - 0x3000,
            ];
            // UTF-16LE "winpeshl" (case-insensitive via lowercase only here; PE paths are lower).
            let needle: [u8; 16] = [
                b'w', 0, b'i', 0, b'n', 0, b'p', 0, b'e', 0, b's', 0, b'h', 0, b'l', 0,
            ];
            for peb in PEB_CANDS {
                // PEB.ProcessParameters @ +0x20
                let Some(pp_pa0) = translate_long(machine, dtb, peb + 0x20) else {
                    continue;
                };
                let pp = u64::from_le_bytes(
                    machine
                        .read_physical_bytes(pp_pa0, 8)
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                // As above: the low bound subsumes the wider comparison it was written with.
                if pp < 0x7000_0000 {
                    continue;
                }
                // RTL_USER_PROCESS_PARAMETERS.ImagePathName UNICODE_STRING @ +0x60
                // { Length:u16, MaxLength:u16, pad:u32, Buffer:u64 }
                let Some(us_pa) = translate_long(machine, dtb, pp + 0x60) else {
                    continue;
                };
                let us = machine.read_physical_bytes(us_pa, 16);
                let len = u16::from_le_bytes([us[0], us[1]]) as usize;
                if len == 0 || len > 512 {
                    continue;
                }
                let buf_ptr = u64::from_le_bytes(us[8..16].try_into().unwrap_or([0; 8]));
                if buf_ptr == 0 {
                    continue;
                }
                let mut path = vec![0u8; len.min(512)];
                let mut ok = true;
                for (i, b) in path.iter_mut().enumerate() {
                    match translate_long(machine, dtb, buf_ptr + i as u64) {
                        Some(pa) => *b = machine.read_physical_bytes(pa, 1)[0],
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                // Case-insensitive ASCII-in-UTF16 match for "winpeshl"
                let lower: Vec<u8> = path
                    .iter()
                    .map(|&c| if c.is_ascii_uppercase() { c + 32 } else { c })
                    .collect();
                if find_bytes(&lower, &needle).is_some() {
                    return true;
                }
            }
            false
        }

        let (pid, dtb, base, cached_gadget) = if let Some(&(pid, dtb, base, g)) = CACHED.get() {
            // g==0 is a sentinel that should not appear; treat as uncached gadget.
            (pid, dtb, base, if g != 0 { Some(g) } else { None })
        } else {
            let tick = MISS_TICKS.fetch_add(1, Ordering::Relaxed);

            // Cheap path every slice: known cold-path winpeshl DTB first, then current CR3.
            // Must pass PEB ImagePathName + WpeUtil IAT (IAT alone false-positive'd services).
            let cur_cr3 = machine.cpu().control.cr3 & !0xfffu64;
            let candidates: [u64; 3] = [0x2a60_1000, 0x2a60_0000, cur_cr3];
            let mut found: Option<(u64, u64, u64)> = None; // pid, dtb, base
            for &cand in &candidates {
                if cand < 0x10_0000 || (cand & 0xfff) != 0 {
                    continue;
                }
                if !peb_path_is_winpeshl(machine, cand) {
                    continue;
                }
                if let Some(base) = resolve_image_base(machine, cand) {
                    if wpeutil_iat_sane(machine, cand, base) {
                        found = Some((0, cand, base));
                        break;
                    }
                }
            }
            if let Some((pid, dtb, base)) = found {
                (pid, dtb, base, None)
            } else {
                // Full EPROCESS scan rarely — every 200 slices (~20M inst @ 100k/slice).
                // Still early enough: winpeshl spends far more than 20M before SCE RPC.
                if !tick.is_multiple_of(200) {
                    bail!("winpeshl.exe not found yet (scan throttled)");
                }
                const OFF_DTB: usize = 0x28;
                const OFF_PID: usize = 0x180;
                const OFF_IMAGE: usize = 0x2e0;
                const OFF_EXIT_TIME: usize = 0x170;
                let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(2048)
                    .saturating_mul(1024 * 1024)
                    .min(4 * 1024 * 1024 * 1024);
                let needle = b"winpeshl.exe";
                let mut winpeshl_dtb: Option<u64> = None;
                let mut winpeshl_pid: Option<u64> = None;
                let chunk = 1024 * 1024usize;
                let mut addr = 0u64;
                while addr < ram_bytes {
                    let len = ((ram_bytes - addr) as usize).min(chunk);
                    let buf = machine.read_physical_bytes(addr, len);
                    let mut start = 0usize;
                    while let Some(rel) = find_bytes(&buf[start..], needle) {
                        let img_off = start + rel;
                        if img_off >= OFF_IMAGE {
                            let eproc_off = img_off - OFF_IMAGE;
                            let dtb = u64::from_le_bytes(
                                buf[eproc_off + OFF_DTB..eproc_off + OFF_DTB + 8]
                                    .try_into()
                                    .unwrap_or([0; 8]),
                            );
                            let pid = u64::from_le_bytes(
                                buf[eproc_off + OFF_PID..eproc_off + OFF_PID + 8]
                                    .try_into()
                                    .unwrap_or([0; 8]),
                            );
                            let phys_eproc = addr + eproc_off as u64;
                            let exit_time = u64::from_le_bytes(
                                machine
                                    .read_physical_bytes(phys_eproc + OFF_EXIT_TIME as u64, 8)
                                    .try_into()
                                    .unwrap_or([0; 8]),
                            );
                            // ImageFileName must start with winpeshl.exe (not a substring hit).
                            let name_ok =
                                img_off + 12 <= buf.len() && &buf[img_off..img_off + 12] == needle;
                            if name_ok
                                && exit_time == 0
                                && dtb != 0
                                && (dtb & 0xfff) == 0
                                && dtb < 0x1_0000_0000
                                && pid <= 0x1_0000
                                && pid % 4 == 0
                                && pid >= 4
                            {
                                winpeshl_dtb = Some(dtb);
                                winpeshl_pid = Some(pid);
                                break;
                            }
                        }
                        start += rel + 1;
                    }
                    if winpeshl_dtb.is_some() {
                        break;
                    }
                    addr += len as u64;
                }
                let dtb = winpeshl_dtb
                    .ok_or_else(|| anyhow!("winpeshl.exe not found in EPROCESS walk"))?;
                let pid = winpeshl_pid.unwrap_or(0);
                let base = resolve_image_base(machine, dtb)
                    .ok_or_else(|| anyhow!("winpeshl ImageBase not found under dtb={dtb:#x}"))?;
                if !wpeutil_iat_sane(machine, dtb, base) {
                    bail!("WpeUtil IAT looks invalid under winpeshl dtb={dtb:#x}");
                }
                (pid, dtb, base, None)
            }
        };

        // Place `xor eax,eax; ret` in RX code — NOT ImageBase+0x7f00 (.rdata/NX →
        // 0xC0000005) and NOT low IAT page ~0x1000 (prior bug wrote gadget at
        // +0x1046 into the import address table and corrupted ntdll/kernel32 slots).
        // Prefer INT3 padding beside MSVC import thunks at ~0xb800 (ff 25 … / cc cc).
        const IAT_RVA: u64 = 0x1240; // WpeUtil IAT start (import desc name→WpeUtil.dll)
        const IAT_SLOTS: u64 = 6;
        // Thunk / .text pad window (avoid 0x1000–0x2000 IAT).
        const THUNK_LO: u64 = 0xb000;
        const THUNK_HI: u64 = 0xc000;
        let gadget_bytes = [0x33u8, 0xc0, 0xc3]; // xor eax,eax; ret

        fn read_va_bytes(machine: &mut Machine, dtb: u64, va: u64, n: usize) -> Option<Vec<u8>> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let pa = translate_long(machine, dtb, va + i as u64)?;
                out.push(machine.read_physical_bytes(pa, 1)[0]);
            }
            Some(out)
        }
        fn write_va_bytes(machine: &mut Machine, dtb: u64, va: u64, bytes: &[u8]) -> Result<()> {
            for (i, b) in bytes.iter().enumerate() {
                let pa = translate_long(machine, dtb, va + i as u64)
                    .ok_or_else(|| anyhow!("unmapped va={:#x}", va + i as u64))?;
                machine.write_physical_u8(pa, *b);
            }
            Ok(())
        }

        let gadget_va = if let Some(g) = cached_gadget {
            // Reject stale cache pointing into the IAT page from older wedges.
            if (g.wrapping_sub(base) >= 0x2000) && (g.wrapping_sub(base) < 0x10000) {
                g
            } else {
                0
            }
        } else {
            0
        };
        let gadget_va = if gadget_va != 0 {
            gadget_va
        } else {
            let mut found: Option<u64> = None;
            // 1) Existing ret-0 only in thunk/.text window (not IAT).
            let mut rva = THUNK_LO;
            while rva + 3 <= THUNK_HI {
                if let Some(b) = read_va_bytes(machine, dtb, base + rva, 3) {
                    if b.as_slice() == gadget_bytes {
                        found = Some(base + rva);
                        break;
                    }
                }
                rva += 1;
            }
            // 2) INT3 padding between import thunks (preferred host for gadget).
            if found.is_none() {
                let mut rva = THUNK_LO;
                while rva + 3 <= THUNK_HI {
                    if let Some(b) = read_va_bytes(machine, dtb, base + rva, 3) {
                        if b.iter().all(|&c| c == 0xcc) {
                            write_va_bytes(machine, dtb, base + rva, &gadget_bytes)?;
                            found = Some(base + rva);
                            break;
                        }
                    }
                    rva += 1;
                }
            }
            // 3) Fixed pad after first jmp [iat] stub (RVA 0xb806 on Win7 winpeshl).
            let g = if let Some(g) = found {
                g
            } else {
                let rva = 0xb806u64;
                write_va_bytes(machine, dtb, base + rva, &gadget_bytes)?;
                base + rva
            };
            let _ = CACHED.set((pid, dtb, base, g));
            g
        };
        // Ensure gadget bytes stay present (sticky).
        write_va_bytes(machine, dtb, gadget_va, &gadget_bytes)?;

        // Redirect each WpeUtil IAT slot to the RX gadget
        let mut patched = 0u32;
        for slot in 0..IAT_SLOTS {
            let iat_va = base + IAT_RVA + slot * 8;
            if let Some(pa0) = translate_long(machine, dtb, iat_va) {
                let cur = u64::from_le_bytes(
                    machine
                        .read_physical_bytes(pa0, 8)
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                if cur == 0 {
                    break;
                }
                if cur == gadget_va {
                    patched += 1;
                    continue;
                }
                write_va_bytes(machine, dtb, iat_va, &gadget_va.to_le_bytes())?;
                patched += 1;
            }
        }
        let msg = format!(
            " pid={pid} dtb={dtb:#x} base={base:#x} gadget={gadget_va:#x} iat_slots={patched}"
        );
        // Sticky path uses `let _ = …` without logging — emit once on first apply.
        if LOGGED_OK
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            eprintln!("AERO_AUTO_SCE_WEDGE: first apply{msg}");
        }
        Ok(msg)
    }

    /// Heuristic scan for Win7 SP1 x64 EPROCESS via ImageFileName @ +0x2e0.
    /// Requires: padded 15-byte name, UniqueProcessId @ +0x180 (×4, ≤0x10000),
    /// DirectoryTableBase @ +0x28 page-aligned and non-zero (filters file-cache hits).
    /// Reports ExitTime @ +0x170 (0 while alive) and ExitStatus @ +0x444 (NTSTATUS
    /// after terminate) — Win7 SP1 x64 offsets from vergilius.
    /// Walk the live `EPROCESS` list from `KPCR.Prcb.CurrentThread`.
    ///
    /// Offsets are Win7 SP1 x64: `gs:[0x188]` CurrentThread, `KTHREAD.Process`
    /// 0x210, `EPROCESS.ActiveProcessLinks` 0x188 / `UniqueProcessId` 0x180 /
    /// `ImageFileName` 0x2e0. Uses kernel GS (MSR `KERNEL_GS_BASE` when RIP is
    /// user). Cheap enough to run every few million instructions.
    fn walk_active_process_image_names(machine: &mut Machine) -> Vec<(u64, [u8; 15], u64)> {
        const GS_CURRENT_THREAD: u64 = 0x188;
        const KTHREAD_PROCESS: u64 = 0x210;
        const EPROCESS_PID: u64 = 0x180;
        const EPROCESS_LINKS: u64 = 0x188;
        const EPROCESS_IMAGE: u64 = 0x2e0;

        let long_mode = (machine.cpu().msr.efer & (1 << 10)) != 0
            && (machine.cpu().control.cr0 & 0x8000_0000) != 0;
        if !long_mode {
            return Vec::new();
        }
        let rip = machine.cpu().rip();
        let kpcr = if rip & (1u64 << 63) != 0 {
            machine.cpu().msr.gs_base
        } else {
            machine.cpu().msr.kernel_gs_base
        };
        if kpcr < 0xffff_8000_0000_0000 {
            return Vec::new();
        }
        let cr3 = machine.cpu().control.cr3;
        let read_u64 = |machine: &mut Machine, va: u64| -> Option<u64> {
            let bytes = read_linear_bytes_cr3(machine, va, 8, true, cr3);
            if bytes.iter().all(|&b| b == 0xfe) {
                return None;
            }
            Some(u64::from_le_bytes(bytes.try_into().ok()?))
        };
        let Some(ethread) = read_u64(machine, kpcr.wrapping_add(GS_CURRENT_THREAD)) else {
            return Vec::new();
        };
        let Some(eprocess) = read_u64(machine, ethread.wrapping_add(KTHREAD_PROCESS)) else {
            return Vec::new();
        };
        if eprocess < 0xffff_8000_0000_0000 {
            return Vec::new();
        }
        let head = eprocess.wrapping_add(EPROCESS_LINKS);
        let mut link = head;
        let mut out = Vec::new();
        let mut seen_links = Vec::new();
        for _ in 0..64 {
            if seen_links.contains(&link) {
                break;
            }
            if link < 0xffff_8000_0000_0000 {
                break;
            }
            seen_links.push(link);
            let proc = link.wrapping_sub(EPROCESS_LINKS);
            let Some(pid) = read_u64(machine, proc.wrapping_add(EPROCESS_PID)) else {
                break;
            };
            let name_bytes =
                read_linear_bytes_cr3(machine, proc.wrapping_add(EPROCESS_IMAGE), 15, true, cr3);
            let name_ok = name_bytes.len() == 15
                && name_bytes[0].is_ascii_alphanumeric()
                && name_bytes.iter().all(|&b| b == 0 || b.is_ascii_graphic());
            if name_ok && pid <= 0x1_0000 && pid % 4 == 0 {
                let mut name = [0u8; 15];
                name.copy_from_slice(&name_bytes);
                out.push((pid, name, proc));
            }
            let Some(next) = read_u64(machine, link) else {
                break;
            };
            if next == head {
                break;
            }
            link = next;
        }
        out
    }

    fn dump_eprocess_image_names(machine: &mut Machine) -> String {
        // Offsets for Windows 7 SP1 x64 EPROCESS (public symbols / vergilius).
        const OFF_DTB: usize = 0x28;
        const OFF_EXIT_TIME: usize = 0x170;
        const OFF_PID: usize = 0x180;
        const OFF_IMAGE: usize = 0x2e0;
        const OFF_EXIT_STATUS: usize = 0x444;
        let ram_bytes: u64 = std::env::var("AERO_SCAN_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
            .min(4 * 1024 * 1024 * 1024);
        let chunk = 1024 * 1024usize;
        let mut out = String::new();
        let mut found = 0usize;
        let mut addr: u64 = 0;
        let thread_filter = std::env::var("AERO_DUMP_THREADS")
            .ok()
            .filter(|name| !name.is_empty());
        let module_filter = std::env::var("AERO_DUMP_MODULES")
            .ok()
            .filter(|name| !name.is_empty());
        while addr < ram_bytes && found < 64 {
            let len = ((ram_bytes - addr) as usize).min(chunk);
            let buf = machine.read_physical_bytes(addr, len);
            // Search the RAM chunk once. The older needle-major loop swept every
            // MiB once per process name, making a 2 GiB process dump take minutes.
            // ImageFileName is exactly 15 bytes on this kernel, and filtering by
            // the known first bytes keeps the common path cheap.
            for (img_off, image) in buf.windows(15).enumerate() {
                if found >= 64 {
                    break;
                }
                if !is_known_eprocess_image_name(image) {
                    continue;
                }
                // Need OFF_IMAGE bytes before ImageFileName within this chunk
                // (or straddling — skip edge cases for this heuristic scan).
                if img_off < OFF_IMAGE {
                    continue;
                }
                let eproc_off = img_off - OFF_IMAGE;
                let dtb = u64::from_le_bytes(
                    buf[eproc_off + OFF_DTB..eproc_off + OFF_DTB + 8]
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                let pid = u64::from_le_bytes(
                    buf[eproc_off + OFF_PID..eproc_off + OFF_PID + 8]
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                let phys_eproc = addr + eproc_off as u64;
                // ExitTime / ExitStatus sit past ImageFileName; the name hit only
                // guarantees bytes through +0x2e0+15. Read the tail via physical
                // loads so we still report status when the object spans chunks.
                let exit_time = u64::from_le_bytes(
                    machine
                        .read_physical_bytes(phys_eproc + OFF_EXIT_TIME as u64, 8)
                        .try_into()
                        .unwrap_or([0; 8]),
                );
                let exit_status = u32::from_le_bytes(
                    machine
                        .read_physical_bytes(phys_eproc + OFF_EXIT_STATUS as u64, 4)
                        .try_into()
                        .unwrap_or([0; 4]),
                );
                let end = image.iter().position(|&b| b == 0).unwrap_or(15);
                let nm = String::from_utf8_lossy(&image[..end]).into_owned();
                // DTB page-aligned, non-zero. Idle is the only pid=0
                // EPROCESS; file-cache dots otherwise match as pid=0.
                let idle = nm.eq_ignore_ascii_case("Idle");
                let system = nm.eq_ignore_ascii_case("System");
                let full_image = nm.len() >= 5
                    && (nm.as_bytes().windows(4).any(|w| {
                        w.eq_ignore_ascii_case(b".exe")
                            || w.eq_ignore_ascii_case(b".com")
                            || w.eq_ignore_ascii_case(b".scr")
                    }));
                // File-cache CBS fragments (`soft-windows-u.`) pass the
                // truncated-dot shape with huge fake PIDs and non-zero
                // ExitTime. Keep dead hits only for a real `*.exe` tag.
                if dtb != 0
                    && (dtb & 0xfff) == 0
                    && dtb < 0x1_0000_0000
                    && ((idle && pid == 0)
                        || (!idle && (4..=0x4000).contains(&pid) && pid % 4 == 0))
                    && (exit_time == 0 || full_image || idle || system)
                {
                    let alive = if exit_time == 0 { "alive" } else { "dead" };
                    out.push_str(&format!(
                        " | pid={pid} name='{nm}' dtb={dtb:#x} eproc_pa={phys_eproc:#x} {alive} exit={exit_status:#x}"
                    ));
                    if thread_filter
                        .as_deref()
                        .is_some_and(|filter| thread_filter_matches(filter, &nm))
                    {
                        out.push_str(&dump_eprocess_threads(machine, phys_eproc));
                    }
                    if module_filter
                        .as_deref()
                        .is_some_and(|filter| nm.eq_ignore_ascii_case(filter))
                    {
                        out.push_str(&dump_eprocess_modules(machine, dtb, phys_eproc));
                    }
                    found += 1;
                }
            }
            addr += len as u64;
        }
        if found == 0 {
            out.push_str(" (no validated EPROCESS candidates)");
        } else {
            out.push_str(&format!(" | count={found}"));
        }
        // RAM-scan can still miss a live process (paged-out ImageFileName,
        // straddling a 1 MiB chunk). The ActiveProcessLinks walk is cheap
        // and reports the exact names Setup uses (`Setup.exe`, `drvinst.exe`).
        let live = walk_active_process_image_names(machine);
        if !live.is_empty() {
            out.push_str(" || LIVE");
            let walk_cr3 = machine.cpu().control.cr3;
            for (pid, name, eprocess_va) in live {
                let end = name.iter().position(|&b| b == 0).unwrap_or(15);
                let nm = String::from_utf8_lossy(&name[..end]);
                out.push_str(&format!(" | pid={pid} name='{nm}' eproc={eprocess_va:#x}"));
                if thread_filter
                    .as_deref()
                    .is_some_and(|filter| thread_filter_matches(filter, &nm))
                {
                    if let Some(phys) = translate_long(machine, walk_cr3, eprocess_va) {
                        out.push_str(&dump_eprocess_threads(machine, phys));
                    } else {
                        out.push_str(" threads=unmapped");
                    }
                }
            }
        }
        out
    }

    /// Walk the native 64-bit PEB loader list for a Win7 SP1 x64 process.
    ///
    /// This is intentionally a bring-up diagnostic rather than a general Windows
    /// introspector. The EPROCESS/PEB/LDR offsets are from the exact public symbols
    /// used by the Win7 Setup investigation:
    /// EPROCESS.Peb=0x338, PEB.Ldr=0x18,
    /// PEB_LDR_DATA.InLoadOrderModuleList=0x10, and
    /// LDR_DATA_TABLE_ENTRY.{DllBase,EntryPoint,SizeOfImage,FullDllName,BaseDllName}
    /// = {0x30,0x38,0x40,0x48,0x58}.
    fn dump_eprocess_modules(machine: &mut Machine, dtb: u64, phys_eprocess: u64) -> String {
        const EPROCESS_PEB: u64 = 0x338;
        const PEB_LDR: u64 = 0x18;
        const LDR_IN_LOAD_ORDER: u64 = 0x10;
        const ENTRY_BYTES: usize = 0x68;

        fn u16_at(bytes: &[u8], offset: usize) -> u16 {
            bytes
                .get(offset..offset + 2)
                .and_then(|v| v.try_into().ok())
                .map(u16::from_le_bytes)
                .unwrap_or(0)
        }

        fn u32_at(bytes: &[u8], offset: usize) -> u32 {
            bytes
                .get(offset..offset + 4)
                .and_then(|v| v.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(0)
        }

        fn u64_at(bytes: &[u8], offset: usize) -> u64 {
            bytes
                .get(offset..offset + 8)
                .and_then(|v| v.try_into().ok())
                .map(u64::from_le_bytes)
                .unwrap_or(0)
        }

        fn user_pointer(value: u64) -> bool {
            (0x1_0000..0x0000_8000_0000_0000).contains(&value)
        }

        fn read_unicode(machine: &mut Machine, dtb: u64, descriptor: &[u8]) -> Option<String> {
            let byte_len = usize::from(u16_at(descriptor, 0));
            let buffer = u64_at(descriptor, 8);
            if byte_len == 0 {
                return Some(String::new());
            }
            if byte_len > 1024 || byte_len % 2 != 0 || !user_pointer(buffer) {
                return None;
            }
            let bytes = read_linear_bytes_cr3(machine, buffer, byte_len, true, dtb);
            if bytes.contains(&0xfe) {
                return None;
            }
            let utf16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            Some(String::from_utf16_lossy(&utf16))
        }

        let peb = u64::from_le_bytes(
            machine
                .read_physical_bytes(phys_eprocess.saturating_add(EPROCESS_PEB), 8)
                .try_into()
                .unwrap_or([0; 8]),
        );
        if !user_pointer(peb) {
            return format!(" modules=<invalid-peb:{peb:#x}>");
        }
        let peb_bytes = read_linear_bytes_cr3(machine, peb, 0x20, true, dtb);
        let ldr = u64_at(&peb_bytes, PEB_LDR as usize);
        if !user_pointer(ldr) {
            return format!(" modules=<invalid-ldr:{ldr:#x} peb={peb:#x}>");
        }

        let list_head = ldr + LDR_IN_LOAD_ORDER;
        let list = read_linear_bytes_cr3(machine, list_head, 16, true, dtb);
        let mut link = u64_at(&list, 0);
        let mut seen = Vec::new();
        let mut modules = Vec::new();

        while link != list_head && user_pointer(link) && !seen.contains(&link) && seen.len() < 128 {
            seen.push(link);
            let entry = read_linear_bytes_cr3(machine, link, ENTRY_BYTES, true, dtb);
            let next = u64_at(&entry, 0);
            let dll_base = u64_at(&entry, 0x30);
            let entry_point = u64_at(&entry, 0x38);
            let size = u32_at(&entry, 0x40);
            let name = read_unicode(machine, dtb, &entry[0x58..0x68])
                .or_else(|| read_unicode(machine, dtb, &entry[0x48..0x58]))
                .unwrap_or_else(|| "<invalid-name>".to_owned());
            modules.push(format!(
                "{name}@{dll_base:#x}+{size:#x}(entry={entry_point:#x})"
            ));
            link = next;
        }

        format!(
            " peb={peb:#x} ldr={ldr:#x} modules={} [{}]",
            modules.len(),
            modules.join(" | ")
        )
    }

    /// Walk the Windows 7 SP1 x64 PnP device-node tree and report matching instance paths.
    ///
    /// Offsets are from public `ntkrnlmp.pdb`. The HDD first-boot kernel is the
    /// ISO install-WIM image (`RSDS 4D0C850D…` age 1): `IopRootDeviceNode` is
    /// `0007:603056` → `.data` RVA `0x2703B0`. LSTAR on that build is
    /// `KiSystemCall64Shadow` (KVASCODE), not `KiSystemCall64`. This boot's
    /// KASLR slide is `0xfffff8000260f000` (MZ-mapped; the 2 MiB-aligned
    /// `0xfffff80002600000` page is not present).
    ///
    /// The older RC#51 boot-WIM ntos used RVA `0x27c5b0` at
    /// `0xfffff8000d20…`. `AERO_WIN7_KERNEL_BASE` / `AERO_IOP_ROOT_RVA`
    /// override discovery; without an RVA override both candidates are tried.
    fn dump_win7_devnodes(machine: &mut Machine, spec: &str) -> String {
        const IOP_ROOT_DEVICE_NODE_RVA: u64 = 0x27_03b0;
        const IOP_ROOT_DEVICE_NODE_RVA_LEGACY: u64 = 0x27_c5b0;
        // First guess for the 2 MiB-radius MZ scan; LSTAR walk still covers
        // the older RC#51 slide if this page is not an ntos image.
        const EXPECTED_KERNEL_BASE: u64 = 0xffff_f800_0260_f000;
        const KERNEL_SCAN_RADIUS: u64 = 0x20_0000;
        const DEVICE_NODE_SIZE: usize = 0x268;
        const OFF_SIBLING: usize = 0x00;
        const OFF_CHILD: usize = 0x08;
        const OFF_PARENT: usize = 0x10;
        const OFF_INSTANCE_PATH: usize = 0x28;
        const OFF_SERVICE_NAME: usize = 0x38;
        const OFF_STATE: usize = 0xe0;
        const OFF_COMPLETION_STATUS: usize = 0x13c;
        const OFF_FLAGS: usize = 0x140;
        const OFF_PROBLEM: usize = 0x148;
        const OFF_RESOURCE_LIST: usize = 0x150;
        const OFF_RESOURCE_LIST_TRANSLATED: usize = 0x158;
        const OFF_RESOURCE_REQUIREMENTS: usize = 0x168;
        const OFF_BOOT_RESOURCES: usize = 0x1d0;
        const OFF_BOOT_RESOURCES_TRANSLATED: usize = 0x1d8;
        const MAX_NODES: usize = 1024;

        fn u16_at(bytes: &[u8], offset: usize) -> u16 {
            bytes
                .get(offset..offset + 2)
                .and_then(|b| b.try_into().ok())
                .map(u16::from_le_bytes)
                .unwrap_or(0)
        }

        fn u32_at(bytes: &[u8], offset: usize) -> u32 {
            bytes
                .get(offset..offset + 4)
                .and_then(|b| b.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(0)
        }

        fn u64_at(bytes: &[u8], offset: usize) -> u64 {
            bytes
                .get(offset..offset + 8)
                .and_then(|b| b.try_into().ok())
                .map(u64::from_le_bytes)
                .unwrap_or(0)
        }

        fn canonical_kernel_pointer(value: u64) -> bool {
            (0xffff_8000_0000_0000..=0xffff_ffff_ffff_ffff).contains(&value)
        }

        fn resource_type_name(resource_type: u8) -> &'static str {
            match resource_type {
                0 => "Null",
                1 => "Port",
                2 => "Interrupt",
                3 => "Memory",
                4 => "Dma",
                5 => "DeviceSpecific",
                6 => "BusNumber",
                7 => "DevicePrivate",
                8 => "AssignedResource",
                9 => "SubAllocateFrom",
                128 => "ConfigData",
                129 => "DevicePrivate",
                130 => "PcCardConfig",
                131 => "MfCardConfig",
                132 => "Connection",
                _ => "Unknown",
            }
        }

        fn decode_cm_resource_list(machine: &mut Machine, cr3: u64, address: u64) -> String {
            if address == 0 {
                return "null".to_owned();
            }
            if !canonical_kernel_pointer(address) {
                return format!("invalid-pointer({address:#x})");
            }
            let header = read_linear_bytes_cr3(machine, address, 4, true, cr3);
            let full_count = u32_at(&header, 0) as usize;
            if !(1..=16).contains(&full_count) {
                return format!("invalid-full-count({full_count})");
            }
            // PnP resource lists are normally tiny. The cap keeps a corrupt guest
            // pointer from turning a diagnostic clean stop into an unbounded read.
            let bytes = read_linear_bytes_cr3(machine, address, 0x4000, true, cr3);
            let mut cursor = 4usize;
            let mut out = format!("CM[{full_count}");
            for full_index in 0..full_count {
                if cursor.checked_add(16).is_none_or(|end| end > bytes.len()) {
                    out.push_str(" truncated-header");
                    break;
                }
                let interface_type = u32_at(&bytes, cursor);
                let bus_number = u32_at(&bytes, cursor + 4);
                let version = u16_at(&bytes, cursor + 8);
                let revision = u16_at(&bytes, cursor + 10);
                let descriptor_count = u32_at(&bytes, cursor + 12) as usize;
                out.push_str(&format!(
                    " full{full_index}(if={interface_type} bus={bus_number} v={version} r={revision} count={descriptor_count})"
                ));
                cursor += 16;
                if descriptor_count > 256 {
                    out.push_str(" invalid-descriptor-count");
                    break;
                }
                for descriptor_index in 0..descriptor_count {
                    if cursor.checked_add(20).is_none_or(|end| end > bytes.len()) {
                        out.push_str(" truncated-descriptor");
                        break;
                    }
                    let resource_type = bytes[cursor];
                    let share = bytes[cursor + 1];
                    let flags = u16_at(&bytes, cursor + 2);
                    out.push_str(&format!(
                        " d{descriptor_index}:{resource_type:#x}/{}(share={share} flags={flags:#x}",
                        resource_type_name(resource_type)
                    ));
                    match resource_type {
                        1 | 3 | 8 | 9 | 128 => {
                            let start = u64_at(&bytes, cursor + 4);
                            let length = u32_at(&bytes, cursor + 12);
                            let end = start.saturating_add(u64::from(length).saturating_sub(1));
                            out.push_str(&format!(
                                " start={start:#x} end={end:#x} len={length:#x}"
                            ));
                        }
                        2 => {
                            let level = u32_at(&bytes, cursor + 4);
                            let vector = u32_at(&bytes, cursor + 8);
                            let affinity = u64_at(&bytes, cursor + 12);
                            out.push_str(&format!(
                                " level={level:#x} vector={vector:#x} affinity={affinity:#x}"
                            ));
                        }
                        4 => {
                            let channel = u32_at(&bytes, cursor + 4);
                            let port = u32_at(&bytes, cursor + 8);
                            out.push_str(&format!(" channel={channel:#x} port={port:#x}"));
                        }
                        5 => {
                            let data_size = u32_at(&bytes, cursor + 4);
                            out.push_str(&format!(" data_size={data_size:#x}"));
                        }
                        6 => {
                            let start = u32_at(&bytes, cursor + 4);
                            let length = u32_at(&bytes, cursor + 8);
                            let end = start.saturating_add(length.saturating_sub(1));
                            out.push_str(&format!(
                                " start={start:#x} end={end:#x} len={length:#x}"
                            ));
                        }
                        _ => {}
                    }
                    out.push(')');
                    cursor += 20;
                }
            }
            out.push(']');
            out
        }

        fn decode_io_resource_requirements(
            machine: &mut Machine,
            cr3: u64,
            address: u64,
        ) -> String {
            if address == 0 {
                return "null".to_owned();
            }
            if !canonical_kernel_pointer(address) {
                return format!("invalid-pointer({address:#x})");
            }
            let header = read_linear_bytes_cr3(machine, address, 32, true, cr3);
            let list_size = u32_at(&header, 0) as usize;
            let interface_type = u32_at(&header, 4);
            let bus_number = u32_at(&header, 8);
            let slot_number = u32_at(&header, 12);
            let alternative_count = u32_at(&header, 28) as usize;
            if !(32..=0x1_0000).contains(&list_size) || alternative_count > 64 {
                return format!(
                    "invalid-requirements(size={list_size:#x} alternatives={alternative_count})"
                );
            }
            let bytes = read_linear_bytes_cr3(machine, address, list_size, true, cr3);
            let mut cursor = 32usize;
            let mut out = format!(
                "REQ[size={list_size:#x} if={interface_type} bus={bus_number} slot={slot_number:#x} alternatives={alternative_count}"
            );
            for alternative_index in 0..alternative_count {
                if cursor.checked_add(8).is_none_or(|end| end > bytes.len()) {
                    out.push_str(" truncated-list");
                    break;
                }
                let version = u16_at(&bytes, cursor);
                let revision = u16_at(&bytes, cursor + 2);
                let descriptor_count = u32_at(&bytes, cursor + 4) as usize;
                out.push_str(&format!(
                    " alt{alternative_index}(v={version} r={revision} count={descriptor_count})"
                ));
                cursor += 8;
                if descriptor_count > 512 {
                    out.push_str(" invalid-descriptor-count");
                    break;
                }
                for descriptor_index in 0..descriptor_count {
                    if cursor.checked_add(32).is_none_or(|end| end > bytes.len()) {
                        out.push_str(" truncated-descriptor");
                        break;
                    }
                    let option = bytes[cursor];
                    let resource_type = bytes[cursor + 1];
                    let share = bytes[cursor + 2];
                    let flags = u16_at(&bytes, cursor + 4);
                    out.push_str(&format!(
                        " d{descriptor_index}:{resource_type:#x}/{}(option={option:#x} share={share} flags={flags:#x}",
                        resource_type_name(resource_type)
                    ));
                    match resource_type {
                        1 | 3 => {
                            let length = u32_at(&bytes, cursor + 8);
                            let alignment = u32_at(&bytes, cursor + 12);
                            let minimum = u64_at(&bytes, cursor + 16);
                            let maximum = u64_at(&bytes, cursor + 24);
                            out.push_str(&format!(
                                " len={length:#x} align={alignment:#x} min={minimum:#x} max={maximum:#x}"
                            ));
                        }
                        2 => {
                            let minimum = u32_at(&bytes, cursor + 8);
                            let maximum = u32_at(&bytes, cursor + 12);
                            out.push_str(&format!(" min={minimum:#x} max={maximum:#x}"));
                        }
                        4 => {
                            let minimum = u32_at(&bytes, cursor + 8);
                            let maximum = u32_at(&bytes, cursor + 12);
                            out.push_str(&format!(" min={minimum:#x} max={maximum:#x}"));
                        }
                        6 => {
                            let length = u32_at(&bytes, cursor + 8);
                            let minimum = u32_at(&bytes, cursor + 12);
                            let maximum = u32_at(&bytes, cursor + 16);
                            out.push_str(&format!(
                                " len={length:#x} min={minimum:#x} max={maximum:#x}"
                            ));
                        }
                        _ => {}
                    }
                    out.push(')');
                    cursor += 32;
                }
            }
            out.push(']');
            out
        }

        fn pe_looks_like_ntos(machine: &mut Machine, cr3: u64, base: u64) -> bool {
            let dos = read_linear_bytes_cr3(machine, base, 0x40, true, cr3);
            if dos.get(..2) != Some(b"MZ") {
                return false;
            }
            let e_lfanew = dos
                .get(0x3c..0x40)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(u32::MAX) as u64;
            if !(0x40..0x400).contains(&e_lfanew) {
                return false;
            }
            let pe = read_linear_bytes_cr3(machine, base + e_lfanew, 0x58, true, cr3);
            if pe.get(..4) != Some(b"PE\0\0")
                || pe.get(4..6) != Some(&0x8664u16.to_le_bytes())
                || pe.get(24..26) != Some(&0x20bu16.to_le_bytes())
            {
                return false;
            }
            let sections = pe
                .get(6..8)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u16::from_le_bytes)
                .unwrap_or(0);
            let image_size = pe
                .get(24 + 0x38..24 + 0x3c)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(0);
            (1..=64).contains(&sections) && (0x40_0000..=0x200_0000).contains(&image_size)
        }

        fn discover_kernel_base(machine: &mut Machine, cr3: u64) -> Option<u64> {
            let start = EXPECTED_KERNEL_BASE.saturating_sub(KERNEL_SCAN_RADIUS) & !0xfff;
            let end = EXPECTED_KERNEL_BASE.saturating_add(KERNEL_SCAN_RADIUS);
            for base in (start..=end).step_by(0x1000) {
                if pe_looks_like_ntos(machine, cr3, base) {
                    return Some(base);
                }
            }
            // HDD first-boot KASLR lands ntos around `0xfffff80002600000`,
            // ~170 MiB away from the old RC#41 slide the fallback still
            // names. Walk 4 KiB pages down from `LSTAR` (KiSystemCall64).
            let lstar = machine.cpu().msr.lstar;
            if lstar >= 0xffff_8000_0000_0000 {
                let mut base = lstar & !0xfff;
                for _ in 0..4096 {
                    if pe_looks_like_ntos(machine, cr3, base) {
                        return Some(base);
                    }
                    if base < 0xffff_8000_0000_1000 {
                        break;
                    }
                    base -= 0x1000;
                }
            }
            None
        }

        fn unicode_string_at(
            machine: &mut Machine,
            cr3: u64,
            node: &[u8],
            offset: usize,
        ) -> String {
            let byte_len = usize::from(u16_at(node, offset));
            let buffer = u64_at(node, offset + 8);
            if byte_len == 0 {
                return String::new();
            }
            if byte_len > 0x400 || byte_len % 2 != 0 || !canonical_kernel_pointer(buffer) {
                return format!("<invalid UNICODE_STRING len={byte_len:#x} buf={buffer:#x}>");
            }
            let bytes = read_linear_bytes_cr3(machine, buffer, byte_len, true, cr3);
            let utf16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            String::from_utf16_lossy(&utf16)
        }

        fn state_name(state: u32) -> &'static str {
            match state {
                0x300 => "Unspecified",
                0x301 => "Uninitialized",
                0x302 => "Initialized",
                0x303 => "DriversAdded",
                0x304 => "ResourcesAssigned",
                0x305 => "StartPending",
                0x306 => "StartCompletion",
                0x307 => "StartPostWork",
                0x308 => "Started",
                0x309 => "QueryStopped",
                0x30a => "Stopped",
                0x30b => "RestartCompletion",
                0x30c => "EnumeratePending",
                0x30d => "EnumerateCompletion",
                0x30e => "AwaitingQueuedDeletion",
                0x30f => "AwaitingQueuedRemoval",
                0x310 => "QueryRemoved",
                0x311 => "RemovePendingCloses",
                0x312 => "Removed",
                0x313 => "DeletePendingCloses",
                0x314 => "Deleted",
                _ => "Unknown",
            }
        }

        let cr3 = std::env::var("AERO_DUMP_CR3")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| machine.cpu().control.cr3);
        let kernel_base_override = std::env::var("AERO_WIN7_KERNEL_BASE")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        let discovered_kernel_base = kernel_base_override
            .is_none()
            .then(|| discover_kernel_base(machine, cr3))
            .flatten();
        let kernel_base = kernel_base_override
            .or(discovered_kernel_base)
            .unwrap_or(EXPECTED_KERNEL_BASE);
        let kernel_base_source = if kernel_base_override.is_some() {
            "override"
        } else if discovered_kernel_base.is_some() {
            "discovered"
        } else {
            "fallback"
        };
        let root_rva_override = std::env::var("AERO_IOP_ROOT_RVA")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        let rva_candidates: Vec<u64> = match root_rva_override {
            Some(rva) => vec![rva],
            None => vec![IOP_ROOT_DEVICE_NODE_RVA, IOP_ROOT_DEVICE_NODE_RVA_LEGACY],
        };
        let mut resolved_root = None;
        let mut last_slot = kernel_base.wrapping_add(IOP_ROOT_DEVICE_NODE_RVA);
        let mut last_root = 0u64;
        let mut last_rva = IOP_ROOT_DEVICE_NODE_RVA;
        for root_rva in rva_candidates {
            let root_slot = kernel_base.wrapping_add(root_rva);
            let root_bytes = read_linear_bytes_cr3(machine, root_slot, 8, true, cr3);
            let root = u64_at(&root_bytes, 0);
            last_slot = root_slot;
            last_root = root;
            last_rva = root_rva;
            if canonical_kernel_pointer(root) {
                resolved_root = Some((root, root_slot, root_rva));
                break;
            }
        }
        let Some((root, root_slot, root_rva)) = resolved_root else {
            return format!(
                " invalid IopRootDeviceNode={last_root:#x} slot={last_slot:#x} rva={last_rva:#x} kernel_base={kernel_base:#x} kernel_base_source={kernel_base_source} cr3={cr3:#x}"
            );
        };

        let filters: Vec<String> = spec
            .split(',')
            .map(str::trim)
            .filter(|filter| !filter.is_empty())
            .map(|filter| filter.to_ascii_lowercase())
            .collect();
        let dump_resources = std::env::var_os("AERO_DUMP_DEVNODE_RESOURCES").is_some();
        let mut stack = vec![(root, 0usize)];
        let mut seen = Vec::new();
        let mut matches = Vec::new();

        while let Some((address, depth)) = stack.pop() {
            if seen.len() >= MAX_NODES || seen.contains(&address) {
                continue;
            }
            seen.push(address);
            let node = read_linear_bytes_cr3(machine, address, DEVICE_NODE_SIZE, true, cr3);
            let sibling = u64_at(&node, OFF_SIBLING);
            let child = u64_at(&node, OFF_CHILD);
            let parent = u64_at(&node, OFF_PARENT);
            let instance = unicode_string_at(machine, cr3, &node, OFF_INSTANCE_PATH);
            let service = unicode_string_at(machine, cr3, &node, OFF_SERVICE_NAME);
            let instance_lower = instance.to_ascii_lowercase();
            if filters.iter().any(|filter| instance_lower.contains(filter)) {
                let state = u32_at(&node, OFF_STATE);
                let completion = u32_at(&node, OFF_COMPLETION_STATUS);
                let flags = u32_at(&node, OFF_FLAGS);
                let problem = u32_at(&node, OFF_PROBLEM);
                let parent_path = if canonical_kernel_pointer(parent) {
                    let parent_node =
                        read_linear_bytes_cr3(machine, parent, DEVICE_NODE_SIZE, true, cr3);
                    unicode_string_at(machine, cr3, &parent_node, OFF_INSTANCE_PATH)
                } else {
                    String::new()
                };
                matches.push(format!(
                    " | depth={depth} node={address:#x} parent={parent:#x} parent_path='{parent_path}' child={child:#x} sibling={sibling:#x} path='{instance}' service='{service}' state={state:#x}({}) completion={completion:#x} problem={problem:#x} flags={flags:#x}",
                    state_name(state)
                ));
                if dump_resources {
                    let resource_list = u64_at(&node, OFF_RESOURCE_LIST);
                    let resource_list_translated = u64_at(&node, OFF_RESOURCE_LIST_TRANSLATED);
                    let resource_requirements = u64_at(&node, OFF_RESOURCE_REQUIREMENTS);
                    let boot_resources = u64_at(&node, OFF_BOOT_RESOURCES);
                    let boot_resources_translated = u64_at(&node, OFF_BOOT_RESOURCES_TRANSLATED);
                    matches.push(format!(
                        " | resources node={address:#x} raw={resource_list:#x} {} translated={resource_list_translated:#x} {} requirements={resource_requirements:#x} {} boot={boot_resources:#x} {} boot_translated={boot_resources_translated:#x} {}",
                        decode_cm_resource_list(machine, cr3, resource_list),
                        decode_cm_resource_list(machine, cr3, resource_list_translated),
                        decode_io_resource_requirements(
                            machine,
                            cr3,
                            resource_requirements
                        ),
                        decode_cm_resource_list(machine, cr3, boot_resources),
                        decode_cm_resource_list(machine, cr3, boot_resources_translated)
                    ));
                }
            }
            if canonical_kernel_pointer(sibling) {
                stack.push((sibling, depth));
            }
            if canonical_kernel_pointer(child) {
                stack.push((child, depth.saturating_add(1)));
            }
        }

        if matches.is_empty() {
            format!(
                " no matching nodes (walked={} root={root:#x} slot={root_slot:#x} rva={root_rva:#x} kernel_base={kernel_base:#x} kernel_base_source={kernel_base_source} cr3={cr3:#x})",
                seen.len()
            )
        } else {
            format!(
                " walked={} root={root:#x} slot={root_slot:#x} rva={root_rva:#x} kernel_base={kernel_base:#x} kernel_base_source={kernel_base_source} cr3={cr3:#x}{}",
                seen.len(),
                matches.concat()
            )
        }
    }

    /// Dump the threads linked from a Win7 SP1 x64 EPROCESS.
    ///
    /// Offsets are from the exact public `ntkrnlmp.pdb` used by this bring-up:
    /// EPROCESS.ThreadListHead=0x308, ActiveThreads=0x328;
    /// ETHREAD.ThreadListEntry=0x420; KTHREAD.State=0x164,
    /// ContextSwitches=0x134, WaitTime=0x194, WaitReason=0x26b.
    fn dump_eprocess_threads(machine: &mut Machine, phys_eprocess: u64) -> String {
        const EPROCESS_THREAD_LIST_HEAD: u64 = 0x308;
        const EPROCESS_ACTIVE_THREADS_FROM_HEAD: usize = 0x20;
        const ETHREAD_THREAD_LIST_ENTRY: u64 = 0x420;
        const KTHREAD_CONTEXT_SWITCHES: usize = 0x134;
        const KTHREAD_STATE: usize = 0x164;
        const KTHREAD_WAIT_TIME: usize = 0x194;
        const KTHREAD_WAIT_REASON: usize = 0x26b;
        const ETHREAD_START_ADDRESS: usize = 0x388;
        const ETHREAD_CID_THREAD: usize = 0x3b8;
        const ETHREAD_WIN32_START_ADDRESS: usize = 0x410;
        const ETHREAD_BYTES: usize = 0x430;

        fn u32_at(bytes: &[u8], offset: usize) -> u32 {
            bytes
                .get(offset..offset + 4)
                .and_then(|v| v.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(0)
        }

        fn u64_at(bytes: &[u8], offset: usize) -> u64 {
            bytes
                .get(offset..offset + 8)
                .and_then(|v| v.try_into().ok())
                .map(u64::from_le_bytes)
                .unwrap_or(0)
        }

        fn state_name(state: u8) -> &'static str {
            match state {
                0 => "Initialized",
                1 => "Ready",
                2 => "Running",
                3 => "Standby",
                4 => "Terminated",
                5 => "Waiting",
                6 => "Transition",
                7 => "DeferredReady",
                8 => "GateWait",
                9 => "ProcessSwapWait",
                _ => "Unknown",
            }
        }

        fn wait_reason_name(reason: u8) -> &'static str {
            match reason {
                0 => "Executive",
                1 => "FreePage",
                2 => "PageIn",
                3 => "PoolAllocation",
                4 => "DelayExecution",
                5 => "Suspended",
                6 => "UserRequest",
                7 => "WrExecutive",
                8 => "WrFreePage",
                9 => "WrPageIn",
                10 => "WrPoolAllocation",
                11 => "WrDelayExecution",
                12 => "WrSuspended",
                13 => "WrUserRequest",
                14 => "WrEventPair",
                15 => "WrQueue",
                16 => "WrLpcReceive",
                17 => "WrLpcReply",
                18 => "WrVirtualMemory",
                19 => "WrPageOut",
                20 => "WrRendezvous",
                25 => "WrCalloutStack",
                26 => "WrKernel",
                27 => "WrResource",
                28 => "WrPushLock",
                29 => "WrMutex",
                30 => "WrQuantumEnd",
                31 => "WrDispatchInt",
                32 => "WrPreempted",
                33 => "WrYieldExecution",
                34 => "WrFastMutex",
                35 => "WrGuardedMutex",
                36 => "WrRundown",
                _ => "Other",
            }
        }

        let list = machine.read_physical_bytes(
            phys_eprocess.saturating_add(EPROCESS_THREAD_LIST_HEAD),
            EPROCESS_ACTIVE_THREADS_FROM_HEAD + 4,
        );
        let active_threads = u32_at(&list, EPROCESS_ACTIVE_THREADS_FROM_HEAD);
        let mut link = u64_at(&list, 0);
        // Session-space ETHREADs (`0xfffffa80…`) are not in the System map.
        // Walk them with the target process DTB, not whatever CR3 the CPU
        // happened to stop on.
        const OFF_DTB: u64 = 0x28;
        let dtb_bytes = machine.read_physical_bytes(phys_eprocess.saturating_add(OFF_DTB), 8);
        let proc_dtb = u64_at(&dtb_bytes, 0) & !0xfff;
        let cr3 = if proc_dtb != 0 {
            proc_dtb
        } else {
            machine.cpu().control.cr3
        };
        let mut out = format!(" threads={active_threads}");
        let mut seen = Vec::new();

        for index in 0..active_threads.min(16) {
            if link < 0xffff_8000_0000_0000 || seen.contains(&link) {
                out.push_str(&format!(" [thread#{index}: invalid_link={link:#x}]"));
                break;
            }
            seen.push(link);
            let Some(ethread) = link.checked_sub(ETHREAD_THREAD_LIST_ENTRY) else {
                out.push_str(&format!(" [thread#{index}: underflow_link={link:#x}]"));
                break;
            };
            let bytes = read_linear_bytes_cr3(machine, ethread, ETHREAD_BYTES, true, cr3);
            if bytes.iter().all(|&byte| byte == 0xfe) {
                out.push_str(&format!(
                    " [thread#{index}: ethread={ethread:#x} unmapped cr3={cr3:#x}]"
                ));
                break;
            }

            let state = bytes[KTHREAD_STATE];
            let wait_reason = bytes[KTHREAD_WAIT_REASON];
            let tid = u64_at(&bytes, ETHREAD_CID_THREAD);
            let context_switches = u32_at(&bytes, KTHREAD_CONTEXT_SWITCHES);
            let wait_time = u32_at(&bytes, KTHREAD_WAIT_TIME);
            let start = u64_at(&bytes, ETHREAD_START_ADDRESS);
            let win32_start = u64_at(&bytes, ETHREAD_WIN32_START_ADDRESS);
            out.push_str(&format!(
                " [tid={tid} ethread={ethread:#x} state={}({state}) wait={}({wait_reason}) ctx={context_switches} wait_time={wait_time} start={start:#x} win32_start={win32_start:#x}]",
                state_name(state),
                wait_reason_name(wait_reason),
            ));
            link = u64_at(&bytes, ETHREAD_THREAD_LIST_ENTRY as usize);
        }
        out
    }

    fn dump_vga_png(machine: &mut Machine, path: &Path) -> Result<()> {
        machine.display_present();
        let (w, h) = machine.display_resolution();
        if w == 0 || h == 0 {
            bail!("no VGA framebuffer available (resolution was {w}x{h})");
        }

        let fb = machine.display_framebuffer();
        let expected_len = (w as usize)
            .checked_mul(h as usize)
            .context("framebuffer size overflow")?;
        if fb.len() != expected_len {
            bail!(
                "unexpected framebuffer length: got {}, expected {} ({w}x{h})",
                fb.len(),
                expected_len
            );
        }

        // `aero_gpu_vga` framebuffer pixels are u32 with little-endian RGBA byte order:
        //   value = R | (G<<8) | (B<<16) | (A<<24)
        // Convert to an explicit RGBA byte buffer for the `image` crate.
        let mut rgba = Vec::with_capacity(fb.len() * 4);
        for &p in fb {
            rgba.push((p & 0xFF) as u8); // R
            rgba.push(((p >> 8) & 0xFF) as u8); // G
            rgba.push(((p >> 16) & 0xFF) as u8); // B
            rgba.push(((p >> 24) & 0xFF) as u8); // A
        }

        let img =
            image::RgbaImage::from_raw(w, h, rgba).ok_or_else(|| anyhow!("invalid image data"))?;
        img.save(path)
            .with_context(|| format!("failed to write PNG: {}", path.display()))?;
        Ok(())
    }

    #[cfg(test)]
    mod type5_wedge_tests {
        use super::{browser_code_to_bios_key, type5_hook_entry_is_bad};

        #[test]
        fn bring_up_keys_have_classic_bios_words() {
            assert_eq!(browser_code_to_bios_key("Enter"), Some(0x1c0d));
            assert_eq!(browser_code_to_bios_key("Space"), Some(0x3920));
            assert_eq!(browser_code_to_bios_key("Escape"), Some(0x011b));
            assert_eq!(browser_code_to_bios_key("Tab"), Some(0x0f09));
            assert_eq!(browser_code_to_bios_key("KeyA"), None);
        }

        #[test]
        fn null_object_is_bad() {
            assert!(type5_hook_entry_is_bad(0, Some(0x1234)));
            assert!(type5_hook_entry_is_bad(0x7, Some(0x1234))); // attributes only
        }

        #[test]
        fn zero_head_h_is_bad() {
            assert!(type5_hook_entry_is_bad(0xffff_f900_c010_4630, Some(0)));
        }

        #[test]
        fn unreadable_head_is_bad() {
            assert!(type5_hook_entry_is_bad(0xffff_f900_c010_4630, None));
        }

        #[test]
        fn healthy_hook_is_kept() {
            assert!(!type5_hook_entry_is_bad(
                0xffff_f900_c010_4630,
                Some(0x0000_0000_0d05_0027)
            ));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}
