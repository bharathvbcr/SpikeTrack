//! This process's physical memory footprint (`task_info(TASK_VM_INFO).phys_footprint`, what
//! Activity Monitor's "Memory" column and memguard read), through the Mach call directly: no
//! dependency. On Apple silicon the footprint includes the GPU buffers this process holds, since
//! they live in the same unified memory.

/// `struct task_vm_info` from `<mach/task_info.h>` up to and including `phys_footprint` (rev1);
/// the kernel fills as many fields as the count passed in says.
#[repr(C)]
#[derive(Default)]
struct TaskVmInfoRev1 {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

/// `TASK_VM_INFO` in `<mach/task_info.h>`.
const TASK_VM_INFO: u32 = 22;

/// `struct vm_statistics64` from `<mach/vm_statistics.h>` through its rev1 fields
/// (`total_uncompressed_pages_in_compressor`), which hold `swapins` and `swapouts`.
#[repr(C)]
#[derive(Default)]
struct VmStatistics64Rev1 {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
}

/// `HOST_VM_INFO64` in `<mach/host_info.h>`.
const HOST_VM_INFO64: i32 = 4;

extern "C" {
    /// `mach_task_self()` is a macro for this global (`<mach/mach_init.h>`).
    static mach_task_self_: u32;
    fn task_info(target_task: u32, flavor: u32, task_info_out: *mut i32, task_info_out_count: *mut u32) -> i32;
    fn mach_host_self() -> u32;
    fn host_statistics64(host_priv: u32, flavor: i32, host_info64_out: *mut i32, host_info64_out_count: *mut u32) -> i32;
}

/// Machine-wide lifetime counts of pages paged out and swapped in (what `vm_stat` prints as
/// Pageouts and Swapins), or `None` if the kernel refused the call or returned fewer fields.
pub fn vm_pageouts_swapins() -> Option<(u64, u64)> {
    // mach_host_self() hands out a send right on every call; take one for the process.
    static HOST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    // SAFETY: no arguments; returns this host's name port.
    let host = *HOST.get_or_init(|| unsafe { mach_host_self() });
    let mut info = VmStatistics64Rev1::default();
    let words = (std::mem::size_of::<VmStatistics64Rev1>() / std::mem::size_of::<u32>()) as u32;
    let mut count = words;
    // SAFETY: `info` is a repr(C) prefix of `vm_statistics64` (whose u64 fields are 8-aligned, as
    // here) and `count` is its size in natural_t words, so the kernel writes at most that much.
    let kr = unsafe { host_statistics64(host, HOST_VM_INFO64, (&mut info as *mut VmStatistics64Rev1).cast::<i32>(), &mut count) };
    (kr == 0 && count == words).then_some((info.pageouts, info.swapins))
}

/// The current physical footprint in bytes, or `None` if the kernel refused the call or returned
/// fewer fields than `phys_footprint` needs.
pub fn phys_footprint_bytes() -> Option<u64> {
    let mut info = TaskVmInfoRev1::default();
    let words = (std::mem::size_of::<TaskVmInfoRev1>() / std::mem::size_of::<u32>()) as u32;
    let mut count = words;
    // SAFETY: `info` is a repr(C) prefix of `task_vm_info` and `count` is its size in natural_t
    // words, so the kernel writes at most that many words into it; `mach_task_self_` is the
    // process's own task port, valid for the process's lifetime.
    let kr = unsafe { task_info(mach_task_self_, TASK_VM_INFO, (&mut info as *mut TaskVmInfoRev1).cast::<i32>(), &mut count) };
    (kr == 0 && count == words).then_some(info.phys_footprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footprint_is_read_and_grows_with_an_allocation() {
        let before = phys_footprint_bytes().expect("task_info(TASK_VM_INFO)");
        assert!(before > 1 << 20, "a running test process has more than 1 MiB of footprint: {before}");
        // 256 MiB, touched so it is resident.
        let block = vec![1u8; 256 << 20];
        let after = phys_footprint_bytes().expect("task_info(TASK_VM_INFO)");
        assert!(after >= before + (200 << 20), "footprint {before} -> {after} after touching 256 MiB");
        assert_eq!(block[block.len() - 1], 1);
    }

    #[test]
    fn vm_counters_match_vm_stat() {
        let (pageouts, swapins) = vm_pageouts_swapins().expect("host_statistics64(HOST_VM_INFO64)");
        let out = std::process::Command::new("/usr/bin/vm_stat").output().expect("vm_stat");
        let text = String::from_utf8_lossy(&out.stdout);
        let field = |name: &str| -> u64 {
            let line = text.lines().find(|l| l.starts_with(name)).unwrap_or_else(|| panic!("no {name} in vm_stat"));
            line.rsplit(' ').next().unwrap().trim_end_matches('.').parse().unwrap()
        };
        // Lifetime counters only grow; vm_stat ran after our read.
        let (po, si) = (field("Pageouts:"), field("Swapins:"));
        assert!(po >= pageouts && po - pageouts < 1 << 20, "pageouts {pageouts} vs vm_stat {po}");
        assert!(si >= swapins && si - swapins < 1 << 20, "swapins {swapins} vs vm_stat {si}");
    }
}
