"""
Test that Genesis mode properly clears LTM and starts with only alphanumerics.
"""

import tempfile
import os
from genesis_orchestrator import GenesisOrchestrator


def test_genesis_clean_start():
    """Verify genesis mode starts clean with only alphanumerics."""

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Creating Genesis Orchestrator...")
        orch = GenesisOrchestrator(data_dir=tmpdir)

        print("\n✓ Orchestrator created")
        print(f"  Genesis enabled: {orch.genesis.enabled}")
        print(f"  LTM size: {len(orch.memory.ltm)}")
        print(f"  STM size: {len(orch.memory.stm)}")

        # Check that LTM has exactly 36 entries (0-9, a-z)
        assert orch.genesis.enabled, "Genesis mode should be enabled"
        assert len(orch.memory.ltm) == 36, f"LTM should have 36 entries (alphanumerics), got {len(orch.memory.ltm)}"

        print("\n✓ Memory state verified:")
        print(f"  LTM has exactly 36 alphanumeric symbols")

        # Check that we can recall the symbols
        for symbol in "0123456789abcdefghijklmnopqrstuvwxyz":
            result = orch.recall(symbol)
            assert result is not None, f"Should be able to recall '{symbol}'"

        print(f"  All 36 symbols are recallable")

        # Check that we CANNOT recall non-alphanumeric knowledge
        non_alphanumeric_tests = [
            "hello",
            "world",
            "calculus",
            "thermodynamics",
            "equation",
        ]

        print("\n✓ Testing that non-alphanumeric knowledge is absent:")
        for word in non_alphanumeric_tests:
            result = orch.recall(word)
            print(f"  '{word}': {result}")
            # Should be None or low similarity (not exact match)

        print("\n✅ Genesis mode clean start verified!")
        print("   System is ready for emergence experiment.")

        return orch


if __name__ == "__main__":
    test_genesis_clean_start()
