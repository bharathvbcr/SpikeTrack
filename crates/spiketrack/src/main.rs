//! `spiketrack`: SAM 3.1 on Apple silicon, one binary, no Python.
//!
//! Commands in this build:
//!
//! * `image`: one image, one text prompt: boxes, scores and masks (see `image_cmd`).
//! * `track`: text-prompted multi-object tracking of a video or frame folder (see `track_cmd`).
//! * `eval-visem`: HOTA of predicted tracks against VISEM-Tracking (see `eval_visem`).

#[cfg(not(target_os = "macos"))]
compile_error!("spiketrack decodes images and video with ImageIO and AVFoundation: macOS only");

mod cli;
mod eval_visem;
mod image_cmd;
mod json;
mod mask;
mod overlay;
mod png;
mod track_cmd;

use std::process::ExitCode;

use cli::CliError;

const USAGE: &str = "\
spiketrack <command> [options]

commands:
  image        detect and segment a text-prompted concept in one image
  track        track a text-prompted concept through a video or a folder of frames
  eval-visem   HOTA of predicted tracks against VISEM-Tracking ground truth

`spiketrack <command> --help` shows a command's options.";

fn dispatch(args: &[String]) -> Result<(), CliError> {
    let Some((command, rest)) = args.split_first() else {
        return Err(CliError::usage("no command given"));
    };
    match command.as_str() {
        "image" => image_cmd::run(rest),
        "track" => track_cmd::run(rest),
        "eval-visem" => eval_visem::run(rest),
        "--help" | "-h" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(CliError::usage(format!("unknown command {other:?}"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(m)) => {
            let usage = match args.first().map(String::as_str) {
                Some("image") => image_cmd::USAGE,
                Some("track") => track_cmd::USAGE,
                Some("eval-visem") => eval_visem::USAGE,
                _ => USAGE,
            };
            eprintln!("spiketrack: {m}\n\n{usage}");
            ExitCode::from(2)
        }
        Err(CliError::Failed(m)) => {
            eprintln!("spiketrack: error: {m}");
            ExitCode::from(1)
        }
    }
}
