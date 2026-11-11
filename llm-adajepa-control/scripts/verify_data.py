"""Verify source/evidence hashes, complete episode grid, and parser replay."""
from audit import load_verified

if __name__ == "__main__":
    _, _, summary = load_verified()
    for key in ("packaged_hashes_verified", "episodes", "response_records",
                "vlm_responses_reparsed", "parser_mismatches"):
        print(f"{key}: {summary[key]}")
    print("PASS: archived evidence verified. Original NPZ receipts are inherited, not reverified.")
