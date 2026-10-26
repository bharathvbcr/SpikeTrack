//! The VISEM loader on the real (git-ignored) dataset:
//! `SPIKETRACK_VISEM=data/visem/VISEM_Tracking_Train_v4/Train cargo test -p st-eval --test visem_data -- --ignored --nocapture`

use std::path::PathBuf;

use st_eval::hota_sequence;
use st_eval::visem::{load_split, VisemClass};

#[test]
#[ignore = "needs the VISEM-Tracking data: set SPIKETRACK_VISEM"]
fn loads_all_twenty_videos_with_track_ids() {
    let dir = PathBuf::from(std::env::var("SPIKETRACK_VISEM").expect("set SPIKETRACK_VISEM"));
    let videos = load_split(&dir).expect("load");
    assert_eq!(videos.len(), 20, "VISEM-Tracking v4 train has 20 annotated videos");
    let (mut frames, mut boxes, mut tracks) = (0usize, 0usize, 0usize);
    for v in &videos {
        let gt = v.hota_frames(&[VisemClass::Sperm]).expect("contiguous frames");
        let ids: std::collections::BTreeSet<&String> = gt.iter().flat_map(|f| f.gt.iter().map(|(id, _)| id)).collect();
        frames += gt.len();
        boxes += gt.iter().map(|f| f.gt.len()).sum::<usize>();
        tracks += ids.len();
        // Ground truth scored against itself is perfect: the loader and HOTA agree on ids.
        let self_frames: Vec<_> = gt.iter().map(|f| st_eval::Frame { gt: f.gt.clone(), pred: f.gt.clone() }).collect();
        let s = hota_sequence(&self_frames).summary();
        assert!((s.hota - 1.0).abs() < 1e-9, "video {}: GT vs itself HOTA {}", v.id, s.hota);
        println!("video {}: {} frames, {} sperm tracks", v.id, gt.len(), ids.len());
    }
    println!("total: {frames} frames, {boxes} sperm boxes, {tracks} sperm tracks");
    assert!(frames > 25_000, "{frames} frames");
}
