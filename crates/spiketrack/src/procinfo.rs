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

extern "C" {
    /// `mach_task_self()` is a macro for this global (`<mach/mach_init.h>`).
    static mach_task_self_: u32;
    fn task_info(target_task: u32, flavor: u32, task_info_out: *mut i32, task_info_out_count: *mut u32) -> i32;
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
}
