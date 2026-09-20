//! A synthetic profile for the flame graph card.
//!
//! Real captures are megabytes, so the gallery grows its own: a deterministic
//! tree of roughly a hundred thousand frames, which is the size the chart's
//! sub-pixel pruning exists for. A card only ever paints the few hundred frames
//! wide enough to see.

use gpui_kit::SharedString;

/// One frame of a sampled stack.
pub struct StackFrame {
    pub label: SharedString,
    /// Samples attributed to this frame and everything it called.
    pub value: f64,
    pub children: Vec<StackFrame>,
}

/// How deep the generated stacks go.
const MAX_DEPTH: usize = 18;
/// Samples in the whole profile, split across the root threads.
const TOTAL_SAMPLES: f64 = 1_800_000.;

/// A reproducible pseudo-random source, so the gallery draws the same profile
/// on every run and a screenshot is comparable across builds.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes' constants: good enough for shaping a demo tree.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// A number in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    /// A fraction in `0.0..1.0`.
    fn fraction(&mut self) -> f64 {
        (self.next() % 10_000) as f64 / 10_000.
    }
}

const VERBS: [&str; 12] = [
    "render", "layout", "paint", "parse", "resolve", "shape", "encode", "hash", "compact",
    "collect", "flush", "poll",
];
const NOUNS: [&str; 10] = [
    "frame", "tree", "buffer", "glyph", "node", "chunk", "index", "span", "event", "slot",
];

/// Build the profile: one frame per thread at the root, each spending its
/// samples down a randomly shaped call tree.
pub fn sample_profile() -> Vec<StackFrame> {
    let mut rng = Lcg(0x5EED);
    let threads = [
        ("main", 0.62),
        ("renderer", 0.24),
        ("worker pool", 0.11),
        ("io", 0.03),
    ];

    threads
        .iter()
        .map(|(name, share)| {
            let value = TOTAL_SAMPLES * share;
            StackFrame {
                label: SharedString::from(*name),
                value,
                children: grow(value, 1, &mut rng),
            }
        })
        .collect()
}

/// Spend some of `parent` on children, leaving the rest as the parent's own
/// time — the uncovered tail a flame graph shows as self time.
///
/// One child at each level is hot, taking most of what is spent. Splitting a
/// parent evenly would halve every width per level and bottom out in sub-pixel
/// frames within a few rows; real profiles have a dominant call chain that
/// stays wide, and that deep spine is the shape a flame graph exists to show.
fn grow(parent: f64, depth: usize, rng: &mut Lcg) -> Vec<StackFrame> {
    if depth >= MAX_DEPTH || parent < 1.5 {
        return vec![];
    }

    let count = 2 + rng.below(5);
    let spend = parent * (0.55 + rng.fraction() * 0.4);
    let hot = rng.below(count);
    let weights: Vec<f64> = (0..count)
        .map(|index| match index == hot {
            true => 4. + rng.fraction() * 6.,
            false => 0.4 + rng.fraction(),
        })
        .collect();
    let total: f64 = weights.iter().sum();

    weights
        .iter()
        .filter_map(|weight| {
            let value = spend * weight / total;
            if value < 1. {
                return None;
            }
            let label = SharedString::from(format!(
                "{}_{}",
                VERBS[rng.below(VERBS.len())],
                NOUNS[rng.below(NOUNS.len())]
            ));
            Some(StackFrame {
                label,
                value,
                children: grow(value, depth + 1, rng),
            })
        })
        .collect()
}

/// How many frames the profile holds, for the card's footnote.
pub fn count(frames: &[StackFrame]) -> usize {
    frames.iter().map(|frame| 1 + count(&frame.children)).sum()
}

/// The deepest stack in the profile.
pub fn depth(frames: &[StackFrame]) -> usize {
    frames
        .iter()
        .map(|frame| 1 + depth(&frame.children))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_profile_is_large_enough_to_exercise_pruning() {
        let profile = sample_profile();
        let frames = count(&profile);

        // The chart's sub-pixel pruning is what this fixture is for, so the
        // gallery's profile has to be the size that needs it.
        assert!(
            (60_000..200_000).contains(&frames),
            "profile has {frames} frames, {} deep",
            depth(&profile)
        );
        assert!(depth(&profile) >= 12);
    }
}
