"""E23 same-tree control (throwaway; nothing ships): patch an exported copy of HEAD so that
`TrackerState::compute_propagation` calls the uncached `MemoryAttention::fwd` on the concatenated
memory again, exactly as at 7f8db18, and nothing else changes. The `MaskMem` cache field stays
(never filled), and so does `prune`'s reset of it.

    python3 -I control_tree.py <exported tree> <7f8db18 state.rs>
"""
import pathlib
import sys

tree, old_state = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
p = tree / "crates/st-model/src/tracker/state.rs"
new = p.read_text()
old = old_state.read_text()


def span(text, start, end):
    """The text from the line holding `start` through the end of the first line holding `end`
    after it; refuses unless each marker occurs once."""
    assert text.count(start) == 1, start
    i = text.index(start)
    i = text.rindex("\n", 0, i) + 1
    j = text.index(end, i)
    j = text.index("\n", j) + 1
    return text[i:j]


cur = span(new, "// Each stored memory brings its projections", "ptrs: ptr_dev.as_ref()")
cur = new[new.index(cur):new.index(")?;", new.index(cur) + len(cur)) + len(")?;\n")]
was = span(old, "// The stored memories, their image features", "num_obj_ptr_tokens: n_ptr,")
was = old[old.index(was):old.index(")?;", old.index(was) + len(was)) + len(")?;\n")]
assert new.count(cur) == 1 and "fwd_frames" in cur and "vos.mem_attn.fwd(" in was
new = new.replace(cur, was)
imp = "use crate::tracker::memory_attention::{FrameKv, FramesInputs};"
assert new.count(imp) == 1
new = new.replace(imp, "use crate::tracker::memory_attention::{FrameKv, FramesInputs, MemoryInputs};")
p.write_text(new)
print(f"replaced {cur.count(chr(10))} lines with {was.count(chr(10))}")
