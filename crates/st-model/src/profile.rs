//! Per-op wall-time profiling of any Ojas backend (for picking the next kernel).
//!
//! [`Profiled`] wraps a backend and forwards every [`Backend`] method, generated from
//! Ojas's own method list (`with_backend_methods!`) so none can be missed. Each `op`
//! method other than `sync` is timed as: call, then `sync` the wrapped backend, then
//! read the clock. That is wall time including dispatch and the synchronization the
//! profiler itself adds; it is not GPU time, and a frame run through it is slower than
//! an unprofiled one. Each call is recorded under the current stage label with its op
//! name and the shapes of its tensor arguments; [`Profiled::report`] aggregates.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ojas_core::{with_backend_methods, Backend, Tensor};

/// One recorded call.
#[derive(Clone, Debug)]
pub struct OpCall {
    pub stage: String,
    pub op: &'static str,
    pub shapes: String,
    pub elapsed: Duration,
}

/// A backend whose op calls are timed and recorded.
pub struct Profiled<B> {
    inner: B,
    stage: Mutex<String>,
    calls: Mutex<Vec<OpCall>>,
}

impl<B: Backend> Profiled<B> {
    pub fn new(inner: B) -> Self {
        Self { inner, stage: Mutex::new("unlabelled".into()), calls: Mutex::new(Vec::new()) }
    }

    /// Label the calls that follow (e.g. "backbone", "detector", "tracker").
    pub fn set_stage(&self, stage: &str) {
        *self.stage.lock().expect("stage lock") = stage.to_string();
    }

    /// Keep only the recorded calls `keep` accepts (e.g. drop a warm-up stage).
    pub fn retain(&self, keep: impl Fn(&OpCall) -> bool) {
        self.calls.lock().expect("calls lock").retain(|c| keep(c));
    }

    pub fn calls(&self) -> Vec<OpCall> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, op: &'static str, shapes: String, elapsed: Duration) {
        let stage = self.stage.lock().expect("stage lock").clone();
        self.calls.lock().expect("calls lock").push(OpCall { stage, op, shapes, elapsed });
    }

    /// Markdown tables: per stage totals, then per (stage, op, shapes) rows sorted by total
    /// time, with call count, total and mean ms, and share of the recorded total.
    pub fn report(&self) -> String {
        let calls = self.calls();
        let total: f64 = calls.iter().map(|c| c.elapsed.as_secs_f64()).sum();
        let mut stages: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        let mut rows: BTreeMap<(&str, &str, &str), (usize, f64)> = BTreeMap::new();
        for c in &calls {
            let s = stages.entry(c.stage.as_str()).or_default();
            s.0 += 1;
            s.1 += c.elapsed.as_secs_f64();
            let r = rows.entry((c.stage.as_str(), c.op, c.shapes.as_str())).or_default();
            r.0 += 1;
            r.1 += c.elapsed.as_secs_f64();
        }
        let mut out = format!("recorded {} op calls, {:.1} ms total (each op synced; wall time)\n\n| stage | calls | ms | share |\n|---|---|---|---|\n", calls.len(), total * 1e3);
        for (stage, (n, t)) in &stages {
            out += &format!("| {stage} | {n} | {:.1} | {:.1}% |\n", t * 1e3, 100.0 * t / total.max(1e-12));
        }
        let mut sorted: Vec<_> = rows.into_iter().collect();
        sorted.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        out += "\n| stage | op | tensor argument shapes | calls | total ms | mean ms | share |\n|---|---|---|---|---|---|---|\n";
        for ((stage, op, shapes), (n, t)) in sorted {
            out += &format!("| {stage} | {op} | {shapes} | {n} | {:.2} | {:.3} | {:.1}% |\n", t * 1e3, t * 1e3 / n as f64, 100.0 * t / total.max(1e-12));
        }
        out
    }
}

/// Shape description of one argument: tensors and optional tensors by shape, attention
/// options by their bias/key-length shapes and flags, everything else omitted.
/// Autoref specialization: the by-value impls win over the blanket `&Arg<T>` fallback.
struct Arg<'a, T>(&'a T);

trait DescribeTensor {
    fn describe(&self) -> Option<String>;
}
impl DescribeTensor for Arg<'_, &Tensor> {
    fn describe(&self) -> Option<String> {
        Some(format!("{:?}", self.0.shape()))
    }
}
impl DescribeTensor for Arg<'_, &mut Tensor> {
    fn describe(&self) -> Option<String> {
        Some(format!("{:?}", self.0.shape()))
    }
}
impl DescribeTensor for Arg<'_, Option<&Tensor>> {
    fn describe(&self) -> Option<String> {
        self.0.map(|t| format!("{:?}", t.shape()))
    }
}
impl DescribeTensor for Arg<'_, ojas_core::AttentionOpts<'_>> {
    fn describe(&self) -> Option<String> {
        let o = self.0;
        let mut parts = Vec::new();
        if o.causal {
            parts.push("causal".to_string());
        }
        if let Some(w) = o.window {
            parts.push(format!("window {w}"));
        }
        if let Some(b) = o.bias {
            parts.push(format!("bias {:?}", b.shape()));
        }
        if let Some(k) = o.key_lens {
            parts.push(format!("key_lens {:?}", k.shape()));
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

trait DescribeOther {
    fn describe(&self) -> Option<String>;
}
impl<T> DescribeOther for &Arg<'_, T> {
    fn describe(&self) -> Option<String> {
        None
    }
}

macro_rules! profile_methods {
    ({ $s:ident } $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
        $( profile_methods!(@one $s, $kind, $name, [$($arg: $ty),*], $ret); )*
    };
    (@one $s:ident, pure, $name:ident, [$($arg:ident : $ty:ty),*], $ret:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(&$s $(, $arg: $ty)*) -> $ret {
            $s.inner.$name($($arg),*)
        }
    };
    (@one $s:ident, op, $name:ident, [$($arg:ident : $ty:ty),*], $ret:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(&$s $(, $arg: $ty)*) -> $ret {
            if stringify!($name) == "sync" {
                return $s.inner.$name($($arg),*);
            }
            let described: Vec<Option<String>> = vec![$( (&Arg(&$arg)).describe() ),*];
            let shapes: Vec<String> = described.into_iter().flatten().collect();
            let t0 = Instant::now();
            let out = $s.inner.$name($($arg),*);
            // Time to completion: a failed sync is the wrapped backend's error to report on
            // its next call, not the profiler's.
            let _ = $s.inner.sync();
            $s.record(stringify!($name), shapes.join(" x "), t0.elapsed());
            out
        }
    };
}

impl<B: Backend> Backend for Profiled<B> {
    with_backend_methods!(profile_methods, { self });
}
