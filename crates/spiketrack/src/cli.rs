//! Hand-rolled argument parsing and the error type shared by the commands.
//!
//! Every option takes a value (`--name value` or `--name=value`); there are no
//! boolean flags besides `--help`. An unknown option, a repeated option, a
//! missing value or a wrong number of positionals is a usage error that names
//! the offender.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// What went wrong, and whether to show the usage text with it.
#[derive(Debug)]
pub enum CliError {
    /// The command line is wrong (exit code 2, usage shown).
    Usage(String),
    /// The command line was fine and the work failed (exit code 1).
    Failed(String),
}

impl CliError {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }

    pub fn failed(msg: impl Into<String>) -> Self {
        Self::Failed(msg.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(m) | Self::Failed(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CliError {}

/// Anything with a `Display` becomes a failure: `.map_err(CliError::from_err)`.
impl CliError {
    pub fn from_err(e: impl fmt::Display) -> Self {
        Self::Failed(e.to_string())
    }
}

/// The parsed command line of one subcommand.
#[derive(Debug, Default)]
pub struct Args {
    pub positional: Vec<String>,
    options: BTreeMap<String, String>,
    pub help: bool,
}

impl Args {
    /// Parse `raw` against `allowed` (option names without the leading `--`) and
    /// exactly `positionals` positional arguments.
    pub fn parse(raw: &[String], allowed: &[&str], positionals: usize) -> Result<Args, CliError> {
        let mut args = Args::default();
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            if a == "--help" || a == "-h" {
                args.help = true;
            } else if let Some(body) = a.strip_prefix("--") {
                if body.is_empty() {
                    return Err(CliError::usage("a bare `--` is not supported"));
                }
                let (name, inline) = match body.split_once('=') {
                    Some((n, v)) => (n, Some(v.to_string())),
                    None => (body, None),
                };
                if !allowed.contains(&name) {
                    return Err(CliError::usage(format!("unknown option --{name} (this command takes: {})", allowed.iter().map(|o| format!("--{o}")).collect::<Vec<_>>().join(", "))));
                }
                let value = match inline {
                    Some(v) => v,
                    None => it.next().cloned().ok_or_else(|| CliError::usage(format!("--{name} needs a value")))?,
                };
                if value.is_empty() {
                    return Err(CliError::usage(format!("--{name} is empty")));
                }
                if args.options.insert(name.to_string(), value).is_some() {
                    return Err(CliError::usage(format!("--{name} given more than once")));
                }
            } else if a.starts_with('-') && a.len() > 1 {
                return Err(CliError::usage(format!("unknown option {a}")));
            } else {
                args.positional.push(a.clone());
            }
        }
        if !args.help && args.positional.len() != positionals {
            return Err(CliError::usage(format!("expected {positionals} positional argument(s), got {}", args.positional.len())));
        }
        Ok(args)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.options.get(name).map(String::as_str)
    }

    pub fn require(&self, name: &str) -> Result<&str, CliError> {
        self.get(name).ok_or_else(|| CliError::usage(format!("--{name} is required")))
    }

    /// A parsed option, or `default` when absent; the error names the option and the text.
    pub fn parsed<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, CliError> {
        match self.get(name) {
            None => Ok(default),
            Some(s) => s.parse().map_err(|_| CliError::usage(format!("--{name}: cannot read {s:?}"))),
        }
    }
}

/// A path from `--name`, else from the environment variable `env`, with a leading
/// `~/` expanded; it must exist.
pub fn existing_path(arg: Option<&str>, env: &str, what: &str, flag: &str) -> Result<PathBuf, CliError> {
    let raw = match arg {
        Some(a) => a.to_string(),
        None => std::env::var(env).map_err(|_| CliError::usage(format!("{what}: pass --{flag} or set {env}")))?,
    };
    let path = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").map_err(|_| CliError::failed(format!("{what}: {raw:?} needs HOME, which is not set")))?).join(rest),
        None => PathBuf::from(&raw),
    };
    if !Path::new(&path).exists() {
        return Err(CliError::failed(format!("{what}: {} does not exist", path.display())));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn options_in_both_spellings_and_positionals() {
        let a = Args::parse(&v(&["img.jpg", "--prompt", "cell", "--threshold=0.3"]), &["prompt", "threshold", "out"], 1).unwrap();
        assert_eq!(a.positional, ["img.jpg"]);
        assert_eq!(a.get("prompt"), Some("cell"));
        assert_eq!(a.parsed("threshold", 0.5f32).unwrap(), 0.3);
        assert_eq!(a.parsed("threads", 8usize).unwrap(), 8);
        assert!(a.get("out").is_none());
    }

    #[test]
    fn usage_errors_name_the_problem() {
        let allowed = ["prompt"];
        let err = |raw: &[&str]| Args::parse(&v(raw), &allowed, 1).unwrap_err().to_string();
        assert!(err(&["a", "--nope", "x"]).contains("unknown option --nope"));
        assert!(err(&["a", "--prompt"]).contains("--prompt needs a value"));
        assert!(err(&["a", "--prompt", "x", "--prompt", "y"]).contains("more than once"));
        assert!(err(&["a", "b"]).contains("expected 1 positional"));
        assert!(err(&[]).contains("expected 1 positional"));
        assert!(err(&["a", "-x"]).contains("unknown option -x"));
        assert!(err(&["a", "--prompt="]).contains("is empty"));
        assert!(err(&["a", "--"]).contains("bare `--`"));
    }

    #[test]
    fn a_bad_number_names_the_option() {
        let a = Args::parse(&v(&["--threshold", "high"]), &["threshold"], 0).unwrap();
        assert!(a.parsed("threshold", 0.5f32).unwrap_err().to_string().contains("--threshold"));
    }

    #[test]
    fn help_skips_the_positional_count() {
        assert!(Args::parse(&v(&["--help"]), &[], 1).unwrap().help);
    }

    #[test]
    fn a_required_option_is_required() {
        let a = Args::parse(&v(&[]), &["pred"], 0).unwrap();
        assert!(matches!(a.require("pred"), Err(CliError::Usage(_))));
    }
}
