"""Tests for _preload_checkpoint_defaults: auto-reading required args from a checkpoint."""
import json
import os
import sys
import pytest


@pytest.fixture(autouse=True)
def restore_argv():
    original = sys.argv[:]
    yield
    sys.argv[:] = original


def make_checkpoint_dir(tmp_path, symmetry="C1", star_fnames=None, particles_dir=None,
                        image_size=160, include_metadata=True, include_config=True):
    """Build a minimal fake checkpoint directory with train_metadata.json and configs_*.yml."""
    ckpt_dir = tmp_path / "version_0"
    ckpt_dir.mkdir()

    if include_metadata:
        metadata = {
            "symmetry": symmetry,
            "particles_star_fname": star_fnames or ["/data/particles.star"],
            "particles_dir": particles_dir,
        }
        (ckpt_dir / "train_metadata.json").write_text(json.dumps(metadata))

    if include_config:
        # Minimal YAML with only the key we care about
        yaml_content = f"datamanager:\n  particlesdataset:\n    image_size_px_for_nnet: {image_size}\n"
        (ckpt_dir / "configs_0.yml").write_text(yaml_content)

    return str(ckpt_dir)


def call_preload(extra_argv=None):
    from cryoPARES.train.train import _preload_checkpoint_defaults
    if extra_argv:
        sys.argv.extend(extra_argv)
    _preload_checkpoint_defaults()


# ---------------------------------------------------------------------------
# continue_checkpoint_dir
# ---------------------------------------------------------------------------

def test_continue_injects_all_missing(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="D7",
                                   star_fnames=["/a.star", "/b.star"])
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir,
                "--n_epochs", "5"]
    call_preload()

    assert "--symmetry" in sys.argv
    assert sys.argv[sys.argv.index("--symmetry") + 1] == "D7"

    assert "--particles_star_fname" in sys.argv
    idx = sys.argv.index("--particles_star_fname")
    assert sys.argv[idx + 1] == "/a.star"
    assert sys.argv[idx + 2] == "/b.star"

    assert "--train_save_dir" in sys.argv
    assert sys.argv[sys.argv.index("--train_save_dir") + 1] == str(tmp_path)

    # Config is loaded directly into main_config, NOT injected into --config
    assert "--config" not in sys.argv


def test_continue_user_symmetry_wins(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="C1")
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir,
                "--symmetry", "C3"]
    call_preload()

    # Should appear exactly once (user's) and equal C3
    sym_indices = [i for i, v in enumerate(sys.argv) if v == "--symmetry"]
    assert len(sym_indices) == 1
    assert sys.argv[sym_indices[0] + 1] == "C3"


def test_continue_user_train_save_dir_wins(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="C1")
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir,
                "--train_save_dir", "/custom/dir"]
    call_preload()

    tsd_indices = [i for i, v in enumerate(sys.argv) if v == "--train_save_dir"]
    assert len(tsd_indices) == 1
    assert sys.argv[tsd_indices[0] + 1] == "/custom/dir"


def test_continue_user_star_fname_wins(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, star_fnames=["/old.star"])
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir,
                "--particles_star_fname", "/new.star"]
    call_preload()

    star_indices = [i for i, v in enumerate(sys.argv) if v == "--particles_star_fname"]
    assert len(star_indices) == 1
    assert sys.argv[star_indices[0] + 1] == "/new.star"


def test_continue_train_save_dir_derived_from_parent(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path)
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir]
    call_preload()

    idx = sys.argv.index("--train_save_dir")
    assert os.path.abspath(sys.argv[idx + 1]) == os.path.abspath(str(tmp_path))


def test_continue_no_config_injected_into_argv(tmp_path):
    # The checkpoint config is loaded directly into main_config, NOT injected into --config.
    # This avoids conflicts when the user also passes direct CLI args like --n_epochs.
    ckpt_dir = make_checkpoint_dir(tmp_path)
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir,
                "--config", "user_override.yaml"]
    original_config_count = sys.argv.count("--config")
    call_preload()
    # --config count must not grow (no extra injection)
    assert sys.argv.count("--config") == original_config_count


# ---------------------------------------------------------------------------
# finetune_checkpoint_dir
# ---------------------------------------------------------------------------

def test_finetune_injects_symmetry_only(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="O")
    sys.argv = ["train.py", "--finetune_checkpoint_dir", ckpt_dir,
                "--particles_star_fname", "/new.star",
                "--train_save_dir", "/new/dir"]
    call_preload()

    assert "--symmetry" in sys.argv
    assert sys.argv[sys.argv.index("--symmetry") + 1] == "O"

    # particles_star_fname must NOT be overridden (user provided /new.star)
    star_idx = sys.argv.index("--particles_star_fname")
    assert sys.argv[star_idx + 1] == "/new.star"


def test_finetune_does_not_inject_train_save_dir(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="C2")
    sys.argv = ["train.py", "--finetune_checkpoint_dir", ckpt_dir]
    call_preload()

    # train_save_dir should NOT be injected for finetune
    assert "--train_save_dir" not in sys.argv


def test_finetune_does_not_inject_particles_star_fname(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, star_fnames=["/old.star"])
    sys.argv = ["train.py", "--finetune_checkpoint_dir", ckpt_dir]
    call_preload()

    assert "--particles_star_fname" not in sys.argv


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_no_checkpoint_flag_noop(tmp_path):
    sys.argv = ["train.py", "--symmetry", "C1", "--n_epochs", "3"]
    original = sys.argv[:]
    call_preload()
    assert sys.argv == original


def test_missing_metadata_file_noop(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, include_metadata=False)
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir]
    call_preload()

    # train_save_dir still injected (derived from path, not metadata)
    assert "--train_save_dir" in sys.argv
    # but symmetry and star fname should not be injected
    assert "--symmetry" not in sys.argv
    assert "--particles_star_fname" not in sys.argv


def test_missing_config_file_noop(tmp_path):
    ckpt_dir = make_checkpoint_dir(tmp_path, include_config=False)
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir]
    call_preload()
    # No --config should be injected; non-config args still work
    assert "--config" not in sys.argv
    assert "--train_save_dir" in sys.argv  # still derived from path


# ---------------------------------------------------------------------------
# Corner cases
# ---------------------------------------------------------------------------

def test_corrupted_metadata_json(tmp_path):
    """Invalid JSON in train_metadata.json must not crash; config and train_save_dir still injected."""
    ckpt_dir = tmp_path / "version_0"
    ckpt_dir.mkdir()
    (ckpt_dir / "train_metadata.json").write_text("{not valid json}")
    yaml_content = "datamanager:\n  particlesdataset:\n    image_size_px_for_nnet: 160\n"
    (ckpt_dir / "configs_0.yml").write_text(yaml_content)

    sys.argv = ["train.py", "--continue_checkpoint_dir", str(ckpt_dir)]
    call_preload()

    # train_save_dir is derived from path, independent of metadata
    assert "--train_save_dir" in sys.argv
    # metadata fields must NOT be injected (parse failed)
    assert "--symmetry" not in sys.argv
    assert "--particles_star_fname" not in sys.argv


def test_partial_metadata_missing_particles_dir(tmp_path):
    """Metadata without 'particles_dir' key must not raise; other fields still injected."""
    ckpt_dir = tmp_path / "version_0"
    ckpt_dir.mkdir()
    metadata = {"symmetry": "C3", "particles_star_fname": ["/data/p.star"]}  # no particles_dir
    (ckpt_dir / "train_metadata.json").write_text(json.dumps(metadata))
    (ckpt_dir / "configs_0.yml").write_text("datamanager:\n  particlesdataset:\n    image_size_px_for_nnet: 160\n")

    sys.argv = ["train.py", "--continue_checkpoint_dir", str(ckpt_dir)]
    call_preload()

    assert "--symmetry" in sys.argv
    assert sys.argv[sys.argv.index("--symmetry") + 1] == "C3"
    assert "--particles_star_fname" in sys.argv
    assert "--particles_dir" not in sys.argv


def test_checkpoint_path_with_trailing_slash(tmp_path):
    """Trailing slash on checkpoint dir must be handled; train_save_dir still resolves to parent."""
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="C1")
    sys.argv = ["train.py", "--continue_checkpoint_dir", ckpt_dir + "/"]
    call_preload()

    assert "--train_save_dir" in sys.argv
    derived = sys.argv[sys.argv.index("--train_save_dir") + 1]
    assert os.path.abspath(derived) == os.path.abspath(str(tmp_path))
    assert "--symmetry" in sys.argv


def test_relative_checkpoint_path(tmp_path, monkeypatch):
    """Relative path for --continue_checkpoint_dir must be converted to absolute."""
    ckpt_dir = make_checkpoint_dir(tmp_path, symmetry="C4")
    monkeypatch.chdir(tmp_path)
    sys.argv = ["train.py", "--continue_checkpoint_dir", "version_0"]
    call_preload()

    assert "--symmetry" in sys.argv
    assert sys.argv[sys.argv.index("--symmetry") + 1] == "C4"
    assert "--train_save_dir" in sys.argv
    derived = sys.argv[sys.argv.index("--train_save_dir") + 1]
    assert os.path.isabs(derived)


def test_multiple_config_files_picks_most_recent(tmp_path):
    """When multiple configs_*.yml exist, the most recently modified one is loaded."""
    import time
    ckpt_dir = tmp_path / "version_0"
    ckpt_dir.mkdir()
    metadata = {"symmetry": "C1", "particles_star_fname": ["/data/p.star"], "particles_dir": None}
    (ckpt_dir / "train_metadata.json").write_text(json.dumps(metadata))

    (ckpt_dir / "configs_0.yml").write_text(
        "datamanager:\n  particlesdataset:\n    image_size_px_for_nnet: 64\n"
    )
    time.sleep(0.05)  # ensure different mtime
    (ckpt_dir / "configs_1.yml").write_text(
        "datamanager:\n  particlesdataset:\n    image_size_px_for_nnet: 160\n"
    )

    from cryoPARES.utils.paths import get_most_recent_file
    chosen = get_most_recent_file(str(ckpt_dir), "configs_*.yml")
    assert chosen.endswith("configs_1.yml"), f"Expected configs_1.yml but got {chosen}"


def test_checkpoint_flag_value_is_another_flag(tmp_path):
    """If --continue_checkpoint_dir is immediately followed by another flag, no injection happens."""
    sys.argv = ["train.py", "--continue_checkpoint_dir", "--symmetry", "C1"]
    original = sys.argv[:]
    call_preload()
    # --symmetry was already in argv; no extra injections should have occurred
    assert sys.argv == original
