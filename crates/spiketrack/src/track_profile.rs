//! `track --profile`: a diagnostic breakdown of each frame by the Session's stages
//! (`st_model::tracker::session::STAGES`) plus the two `track` adds around them ("decode +
//! upload" before a frame, "collector" after it). Not a stable interface.
//!
//! * `stages`: wall time per stage, the Metal device synced as each stage begins, so a stage's
//!   time is the host work it did plus the GPU work it queued. The syncs cost a few waits per
//!   frame and stop work in one stage from overlapping the next: read the stage sum against an
//!   unprofiled run of the same frames before taking the shares literally.
//! * `ops`: every backend op timed through `st_model::profile::Profiled` (a sync after each op,
//!   so much slower than a real frame), labelled with the stage it ran in; frame 0 (two visits,
//!   every birth) is labelled apart.

use std::cell::RefCell;
use std::time::Instant;

use ojas_core::Backend;
use ojas_metal::MetalBackend;
use st_model::profile::Profiled;

use crate::cli::CliError;

/// `--profile MODE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileMode {
    Stages,
    Ops,
}

impl ProfileMode {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "stages" => Ok(Self::Stages),
            "ops" => Ok(Self::Ops),
            other => Err(CliError::usage(format!(
                "--profile: {other:?} is not stages or ops"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Stages => "stages",
            Self::Ops => "ops",
        }
    }
}

/// The stage `track` marks before decoding and uploading a frame, and after the Session returns
/// (the progress line's telemetry, then the collector's masks, boxes and overlays).
pub const DECODE: &str = "decode + upload";
pub const COLLECTOR: &str = "progress line + collector";

/// Where `track` reports stage boundaries.
pub trait StageSink {
    /// `stage` begins (and the previous one, if any, ends).
    fn enter(&self, stage: &str);
    /// The frame is done: the open stage ends. Reports a failure `enter` could not.
    fn end_frame(&self) -> Result<(), CliError>;
}

/// One closed stage: (frame index, stage, seconds).
pub type StageRow = (usize, String, f64);

#[derive(Default)]
struct Clock {
    frame: usize,
    open: Option<(String, Instant)>,
    rows: Vec<StageRow>,
    error: Option<String>,
}

/// `--profile stages`.
pub struct StageClock {
    metal: MetalBackend,
    clock: RefCell<Clock>,
}

impl StageClock {
    pub fn new(metal: &MetalBackend) -> Self {
        Self {
            metal: metal.clone(),
            clock: RefCell::new(Clock::default()),
        }
    }

    fn close(&self, c: &mut Clock) {
        if let Err(e) = self.metal.sync() {
            c.error
                .get_or_insert_with(|| format!("--profile stages: sync: {e}"));
        }
        if let Some((stage, t0)) = c.open.take() {
            c.rows.push((c.frame, stage, t0.elapsed().as_secs_f64()));
        }
    }

    /// The closed stages. A stage still open (the decode that found no next frame) is dropped.
    pub fn rows(&self) -> Vec<StageRow> {
        self.clock.borrow().rows.clone()
    }
}

impl StageSink for StageClock {
    fn enter(&self, stage: &str) {
        let mut c = self.clock.borrow_mut();
        self.close(&mut c);
        c.open = Some((stage.to_string(), Instant::now()));
    }

    fn end_frame(&self) -> Result<(), CliError> {
        let mut c = self.clock.borrow_mut();
        self.close(&mut c);
        c.frame += 1;
        match c.error.take() {
            Some(e) => Err(CliError::failed(e)),
            None => Ok(()),
        }
    }
}

/// `--profile ops`: labels the `Profiled` backend's calls with the stage, frame 0 apart.
pub struct OpsLabels<'p> {
    profiled: &'p Profiled<MetalBackend>,
    frame: RefCell<usize>,
}

impl<'p> OpsLabels<'p> {
    pub fn new(profiled: &'p Profiled<MetalBackend>) -> Self {
        Self {
            profiled,
            frame: RefCell::new(0),
        }
    }
}

impl StageSink for OpsLabels<'_> {
    fn enter(&self, stage: &str) {
        let label = if *self.frame.borrow() == 0 {
            format!("frame 0: {stage}")
        } else {
            stage.to_string()
        };
        self.profiled.set_stage(&label);
    }

    fn end_frame(&self) -> Result<(), CliError> {
        *self.frame.borrow_mut() += 1;
        Ok(())
    }
}

/// The waits and dispatches `Profiled` records per call.
pub fn metal_counters(m: &MetalBackend) -> (u64, u64) {
    (m.waits(), m.dispatches())
}

/// Markdown: frame 0 per stage, then frames 1.. per stage (total, mean per frame, share of the
/// stage sum), stages in the order they first ran; then the stage sum per steady frame against
/// `frame_seconds` (the Session's push_frame time per frame, which excludes decode and collector);
/// then every row as a fenced `tsv` block (frame, stage, ms), so a reader can pick its own frames
/// (the memory bank fills over the first frames, and the means above include those).
pub fn stage_report(rows: &[StageRow], frame_seconds: &[f64]) -> String {
    let mut order: Vec<&str> = Vec::new();
    for (_, s, _) in rows {
        if !order.contains(&s.as_str()) {
            order.push(s);
        }
    }
    let frames = rows.iter().map(|r| r.0 + 1).max().unwrap_or(0);
    let steady = frames.saturating_sub(1);
    // `+ 0.0`: an empty f64 sum is -0.0, which would print as "-0.0".
    let sum = |stage: &str, first: bool| -> f64 {
        rows.iter()
            .filter(|r| r.1 == stage && (r.0 == 0) == first)
            .map(|r| r.2)
            .sum::<f64>()
            + 0.0
    };
    let total_steady: f64 = rows.iter().filter(|r| r.0 > 0).map(|r| r.2).sum();
    let mut out = format!("{frames} frame(s): frame 0 apart (two visits, every birth), then {steady} steady frame(s).\n\n");
    out.push_str("| stage | frame 0 ms | frames 1.. total ms | mean ms / frame | share |\n|---|---|---|---|---|\n");
    for s in &order {
        let st = sum(s, false);
        let share = if total_steady > 0.0 {
            100.0 * st / total_steady
        } else {
            0.0
        };
        let mean = if steady > 0 {
            1e3 * st / steady as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "| {s} | {:.1} | {:.1} | {mean:.1} | {share:.1}% |\n",
            1e3 * sum(s, true),
            1e3 * st
        ));
    }
    let in_session: f64 = rows
        .iter()
        .filter(|r| r.0 > 0 && r.1 != DECODE && r.1 != COLLECTOR)
        .map(|r| r.2)
        .sum();
    let measured: f64 = frame_seconds.iter().skip(1).sum();
    if steady > 0 {
        out.push_str(&format!(
            "\nSteady frames: stage sum {:.1} ms / frame ({:.1} ms inside the Session), push_frame measured {:.1} ms / frame.\n",
            1e3 * total_steady / steady as f64,
            1e3 * in_session / steady as f64,
            1e3 * measured / steady as f64
        ));
    }
    out.push_str("\nPer frame:\n\n```tsv\nframe\tstage\tms\n");
    for (f, s, secs) in rows {
        out.push_str(&format!("{f}\t{s}\t{:.3}\n", 1e3 * secs));
    }
    out.push_str("```\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_modes_parse_and_nothing_else() {
        assert_eq!(ProfileMode::parse("stages").unwrap(), ProfileMode::Stages);
        assert_eq!(ProfileMode::parse("ops").unwrap().name(), "ops");
        for bad in ["", "Stages", "op", "all"] {
            assert!(ProfileMode::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_report_keeps_frame_0_apart_and_sums_the_steady_frames() {
        let r = |f: usize, s: &str, secs: f64| (f, s.to_string(), secs);
        let rows = vec![
            r(0, DECODE, 0.5),
            r(0, "trunk", 2.0),
            r(0, "births", 3.0),
            r(1, DECODE, 0.1),
            r(1, "trunk", 1.0),
            r(1, COLLECTOR, 0.1),
            r(2, DECODE, 0.1),
            r(2, "trunk", 0.6),
            r(2, COLLECTOR, 0.1),
        ];
        let rep = stage_report(&rows, &[5.0, 1.1, 0.7]);
        assert!(rep.starts_with("3 frame(s): frame 0 apart"), "{rep}");
        assert!(
            rep.contains("| trunk | 2000.0 | 1600.0 | 800.0 | 80.0% |"),
            "{rep}"
        );
        assert!(
            rep.contains("| births | 3000.0 | 0.0 | 0.0 | 0.0% |"),
            "{rep}"
        );
        assert!(
            rep.contains("| decode + upload | 500.0 | 200.0 | 100.0 | 10.0% |"),
            "{rep}"
        );
        assert!(rep.contains("stage sum 1000.0 ms / frame (800.0 ms inside the Session), push_frame measured 900.0 ms / frame"), "{rep}");
        let order: Vec<&str> = rep
            .lines()
            .filter(|l| l.starts_with("| ") && !l.starts_with("| stage"))
            .map(|l| l.split(" | ").next().unwrap())
            .collect();
        assert_eq!(
            order,
            [
                "| decode + upload",
                "| trunk",
                "| births",
                "| progress line + collector"
            ]
        );
        let tsv: Vec<&str> = rep
            .lines()
            .skip_while(|l| *l != "```tsv")
            .skip(1)
            .take_while(|l| *l != "```")
            .collect();
        assert_eq!(tsv.len(), 1 + rows.len(), "{rep}");
        assert_eq!(tsv[0], "frame\tstage\tms");
        assert_eq!(tsv[2], "0\ttrunk\t2000.000");
        assert_eq!(tsv[9], "2\tprogress line + collector\t100.000");
    }
}
