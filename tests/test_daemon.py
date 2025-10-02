
import os
import subprocess
import sys
import time
import pytest
from pathlib import Path
import signal
import torch
import yaml

from cryoPARES.inference.daemon.materializePartialResults import materialize_partial_results


@pytest.fixture(scope="module")
def cryo_em_data(tmp_path_factory):
    data_dir = os.path.expanduser("~/tmp/cryoSupervisedDataset")
    if not os.path.isdir(data_dir):
        data_dir = tmp_path_factory.mktemp("daemon_data")
    try:
        from cesped.particlesDataset import ParticlesDataset
        print("CESPED entries", ParticlesDataset.getLocallyAvailableEntries())
        ps = ParticlesDataset("TEST", halfset=1, benchmarkDir=str(data_dir))
    except ImportError:
        pytest.skip("cesped package not found, skipping daemon test. Please install it with 'pip install cesped'")
    except Exception as e:
        pytest.fail(f"Failed to download test data with cesped: {e}")

    # cesped downloads everything into a 'test' subdir
    data_path = os.path.join(data_dir, "TEST")

    # Create a subset with only 30 particles for faster testing
    subset_dir = tmp_path_factory.mktemp("daemon_data_subset")
    import starfile
    import shutil
    star_files = list(Path(data_path).glob("*.star"))
    if star_files:
        data = starfile.read(star_files[0])
        # Handle both dict (multi-table) and dataframe (single table)
        particles_df = None
        if isinstance(data, dict):
            # Multi-table star file - subset the particles table
            for key in data:
                if 'particles' in key.lower():
                    data[key] = data[key].head(30)
                    particles_df = data[key]
                    break
        else:
            data = data.head(30)
            particles_df = data

        subset_star = subset_dir / "particles_subset.star"
        starfile.write(data, subset_star)

        # Copy the referenced mrcs files
        if particles_df is not None and 'rlnImageName' in particles_df.columns:
            mrcs_files = particles_df['rlnImageName'].str.split('@').str[1].unique()
            for mrcs in mrcs_files:
                src = Path(data_path) / mrcs
                if src.exists():
                    shutil.copy(src, subset_dir / mrcs)

    return str(subset_dir)

@pytest.fixture(scope="module")
def dummy_checkpoint(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.mktemp("daemon_checkpoint")

    # The inferencer expects a specific directory structure
    model_dir = checkpoint_dir / "half1" / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy hparams.yaml file
    with open(checkpoint_dir / "half1" / "hparams.yaml", "w") as f:
        yaml.dump({"symmetry": "C1"}, f)

    # Create a dummy config file in the root of the checkpoint dir
    config = {
        "inference": {
            "batch_size": 2,
            "top_k_poses_nnet": 1,
            "directional_zscore_thr": None, # So we don't filter out anything
            "skip_localrefinement": False,
            "skip_reconstruction": False,
            "update_progressbar_n_batches": 1,
        },
                    "datamanager": {
                        "num_dataworkers": 0,
                        "pin_memory": False,
                        "particlesDataset": {                "sampling_rate_angs_for_nnet": 4.0,
                "image_size_px_for_nnet": 64,
            }
        },
        "models": {
            "image2sphere": {
                "lmax": 2
            }
        }
    }
    with open(checkpoint_dir / "configs_dummy.yml", "w") as f:
        yaml.dump(config, f)

    # Create a dummy model script
    model_path = model_dir / "best_script.pt"
    from tests.dummy_model import DummyModel
    torch.jit.script(DummyModel()).save(model_path)

    # Create a dummy directional normalizer
    from cryoPARES.models.directionalNormalizer.directionalNormalizer import DirectionalPercentileNormalizer
    from scipy.spatial.transform import Rotation
    normalizer = DirectionalPercentileNormalizer(symmetry="C1")
    normalizer.fit(torch.FloatTensor(Rotation.random(100).as_matrix()), torch.rand(100))
    torch.save(normalizer, model_dir / "best_directional_normalizer.pt")


    return checkpoint_dir


@pytest.fixture(scope="module")
def reference_map(cryo_em_data):
    # Use a fixed path in /tmp for caching
    cache_dir = Path("/tmp/cryopares_test_cache")
    cache_dir.mkdir(exist_ok=True)
    reference_map_path = cache_dir / "reference.mrc"

    if reference_map_path.exists():
        print(f"Using cached reference map: {reference_map_path}")
        return reference_map_path

    # Find a star file in the cryo_em_data directory
    star_files = list(Path(cryo_em_data).glob("*.star"))
    if not star_files:
        pytest.fail("No star files found in the test dataset.")
    
    star_file = star_files[0]

    # Run the reconstruction script
    from cryoPARES.reconstruction.reconstruct import reconstruct_starfile
    print(f"Generating reference map at: {reference_map_path}")
    reconstruct_starfile(
        particles_star_fname=str(star_file),
        symmetry="C1",
        output_fname=str(reference_map_path),
        n_jobs=1, # Use a single job for simplicity in the test
        use_cuda=torch.cuda.is_available()
    )

    return reference_map_path


def find_free_port(start_port=50000, max_attempts=10):
    """Find a free port by trying to bind to it"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")

def cleanup_port(port):
    """Kill any process listening on the given port"""
    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
            shell=True, capture_output=True
        )
    except Exception:
        pass

def test_daemon_mode(cryo_em_data, dummy_checkpoint, reference_map, tmp_path):
    results_dir = tmp_path / "daemon_results"
    results_dir.mkdir()

    # Find a free port and cleanup any existing processes
    test_port = find_free_port()
    cleanup_port(test_port)

    try:
        # 1. Start Queue Manager in a subprocess
        queue_process = subprocess.Popen(
            [sys.executable, "-m", "cryoPARES.inference.daemon.queueManager",
             "--ip", "localhost", "--port", str(test_port), "--authkey", "test_key"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(2)  # Give it time to start

        # 2. Start Worker in a subprocess
        worker_cmd = [
            sys.executable, "-m", "cryoPARES.inference.daemon.daemonInference",
            "--checkpoint_dir", str(dummy_checkpoint),
            "--results_dir", str(results_dir),
            "--reference_map", str(reference_map),
            "--net_address", "localhost",
            "--net_port", str(test_port),
            "--net_authkey", "test_key",
            "--model_halfset", "half1",
            "--batch_size", "2",
            "--num_dataworkers", "0",
            "--secs_between_partial_results_written", "2",
            "--NOT_use_cuda",  # Force CPU to avoid device mismatch issues in test
            "--config", "inference.skip_localrefinement=False", "inference.skip_reconstruction=False"
        ]

        worker_process = subprocess.Popen(
            worker_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(5)  # Give it time to initialize the model

        # 3. Start Spooler in a subprocess
        spooler_process = subprocess.Popen(
            [sys.executable, "-m", "cryoPARES.inference.daemon.spoolingFiller",
             "--directory", str(cryo_em_data),
             "--ip", "localhost", "--port", str(test_port), "--authkey", "test_key",
             "--pattern", "*.star", "--check_interval", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Let the system run for a bit
        # The spooler should find the files, the worker should process them,
        # and partial results should be written.
        time.sleep(15)

        # 4. Terminate the processes
        # Send poison pill to the queue to stop the worker gracefully
        from cryoPARES.inference.daemon.queueManager import connect_to_queue
        q, m = connect_to_queue(ip="localhost", port=test_port, authkey="test_key")
        q.put(None)
        # The client-side manager does not have a shutdown method.
        # Rely on garbage collection to clean up the connection.

        # Terminate spooler and queue manager
        # We need to use SIGINT to trigger the graceful shutdown
        spooler_process.send_signal(signal.SIGINT)
        try:
            spooler_stdout, spooler_stderr = spooler_process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            spooler_process.kill()
            spooler_stdout, spooler_stderr = spooler_process.communicate()

        # Worker should exit after receiving poison pill
        try:
            worker_stdout, worker_stderr = worker_process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            worker_process.kill()
            worker_stdout, worker_stderr = worker_process.communicate()

        queue_process.send_signal(signal.SIGINT)
        try:
            queue_stdout, queue_stderr = queue_process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            queue_process.kill()
            queue_stdout, queue_stderr = queue_process.communicate()

        print("=== Queue Manager ===")
        if queue_stdout:
            print("stdout:", queue_stdout.decode())
        if queue_stderr:
            print("stderr:", queue_stderr.decode())

        print("\n=== Worker ===")
        if worker_stdout:
            print("stdout:", worker_stdout.decode())
        if worker_stderr:
            print("stderr:", worker_stderr.decode())

        print("\n=== Spooler ===")
        if spooler_stdout:
            print("stdout:", spooler_stdout.decode())
        if spooler_stderr:
            print("stderr:", spooler_stderr.decode())

        # 5. Verify the results
        # Check if partial reconstruction files were created
        partial_results = list(results_dir.glob("mapcomponents_*.npz"))
        assert len(partial_results) > 0, "No partial reconstruction files were created"
        # 6. Materialize the final volume
        output_mrc = results_dir / "final_map.mrc"
        materialize_partial_results(input_files=[str(p) for p in partial_results], output_mrc=str(output_mrc))

        assert output_mrc.exists(), "Final MRC map was not created"
        assert output_mrc.stat().st_size > 0, "Final MRC map is empty"

        # Check for the star file with the results
        result_star_files = list(results_dir.glob("cryoPARES_poses*.star"))
        assert len(result_star_files) > 0, "No result star file was created"

    finally:
        # Cleanup: make sure port is freed
        cleanup_port(test_port)
