#!/usr/bin/env python3
"""Quick integration test for optimized data processing."""

import sys
import tempfile
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from go_rl.sgf_utils import iter_sgf_samples


def create_test_sgf(path: Path, board_size: int = 9, num_moves: int = 10) -> None:
    """Create a minimal test SGF file."""
    sgf_content = f"(;GM[1]FF[4]SZ[{board_size}]"
    for i in range(num_moves):
        if i % 2 == 0:
            sgf_content += f";B[aa]"
        else:
            sgf_content += f";W[bb]"
    sgf_content += ")"

    path.write_text(sgf_content)


def test_parallel_processing():
    """Test parallel SGF processing."""
    print("Testing parallel SGF processing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test SGF files
        num_files = 10
        for i in range(num_files):
            sgf_file = tmpdir / f"test_{i:03d}.sgf"
            create_test_sgf(sgf_file, board_size=9, num_moves=5)

        print(f"Created {num_files} test SGF files")

        # Test sequential processing
        print("\n1. Testing sequential processing...")
        samples_seq = list(
            iter_sgf_samples(str(tmpdir), max_positions=100, use_parallel=False)
        )
        print(f"   Sequential: collected {len(samples_seq)} samples")

        # Test parallel processing
        print("\n2. Testing parallel processing...")
        samples_par = list(
            iter_sgf_samples(str(tmpdir), max_positions=100, use_parallel=True, num_workers=4)
        )
        print(f"   Parallel: collected {len(samples_par)} samples")

        # Verify same number of samples
        assert len(samples_seq) == len(samples_par), (
            f"Sample count mismatch: seq={len(samples_seq)}, par={len(samples_par)}"
        )

        # Verify sample structure
        for sample in samples_seq[:3]:
            assert "prompt" in sample, "Missing 'prompt' field"
            assert "target" in sample, "Missing 'target' field"
            assert "player" in sample, "Missing 'player' field"
            assert "board_size" in sample, "Missing 'board_size' field"
            assert sample["board_size"] == 9, f"Wrong board size: {sample['board_size']}"

        print("\n3. Verifying sample structure...")
        print(f"   Sample keys: {list(samples_seq[0].keys())}")
        print(f"   Board size: {samples_seq[0]['board_size']}")
        print(f"   Player: {samples_seq[0]['player']}")
        print(f"   Target: {samples_seq[0]['target']}")

        print("\n✓ All tests passed!")
        print(f"✓ Parallel processing works correctly")
        print(f"✓ Output format is correct")


def test_buffered_writing():
    """Test buffered JSONL writing."""
    print("\n\nTesting buffered I/O...")

    from go_rl.data_prep import build_sft_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test SGF files
        num_files = 5
        for i in range(num_files):
            sgf_file = tmpdir / f"test_{i:03d}.sgf"
            create_test_sgf(sgf_file, board_size=9, num_moves=3)

        output_file = tmpdir / "output.jsonl"

        # Build dataset
        print(f"Building dataset from {num_files} files...")
        count = build_sft_dataset(
            str(tmpdir),
            str(output_file),
            max_positions=100,
            use_parallel=True,
            batch_size=10,
        )

        print(f"Wrote {count} samples to {output_file}")

        # Verify output
        assert output_file.exists(), "Output file not created"

        lines = output_file.read_text().strip().split("\n")
        print(f"Output has {len(lines)} lines")

        # Verify JSONL format
        import json

        for i, line in enumerate(lines[:3]):
            record = json.loads(line)
            assert "prompt" in record, f"Line {i}: missing 'prompt'"
            assert "response" in record, f"Line {i}: missing 'response'"
            assert "board_size" in record, f"Line {i}: missing 'board_size'"

        print("✓ Buffered I/O works correctly")
        print("✓ JSONL output format is valid")


if __name__ == "__main__":
    try:
        test_parallel_processing()
        test_buffered_writing()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
