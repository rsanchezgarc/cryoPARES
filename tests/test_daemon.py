
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

    # Create a subset with only 10 particles for faster testing
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
                    data[key] = data[key].head(10)
                    particles_df = data[key]
                    break
        else:
            data = data.head(10)
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
        # Get PIDs listening on the port
        result = subprocess.run(
            f"lsof -ti:{port}",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    # Kill each PID silently
                    subprocess.run(
                        f"kill -9 {pid}",
                        shell=True, capture_output=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                    )
                except Exception:
                    pass
    except Exception:
        pass

def test_daemon_mode(cryo_em_data, dummy_checkpoint, reference_map, tmp_path):
    results_dir = tmp_path / "daemon_results"
    results_dir.mkdir()

    # Find a free port and cleanup any existing processes
    test_port = find_free_port()
    cleanup_port(test_port)

    queue_process = None
    worker_process = None
    spooler_process = None

    try:
        # 1. Start Queue Manager in a subprocess
        queue_process = subprocess.Popen(
            [sys.executable, "-m", "cryoPARES.inference.daemon.queueManager",
             "--ip", "localhost", "--port", str(test_port), "--authkey", "test_key"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        time.sleep(2)  # Give it time to start

        # Check if queue manager started successfully
        if queue_process.poll() is not None:
            stdout, stderr = queue_process.communicate()
            pytest.fail(f"Queue manager failed to start:\nstdout: {stdout}\nstderr: {stderr}")

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
            "--config", "inference.skip_localrefinement=True", "inference.skip_reconstruction=True"  # Skip for speed
        ]

        worker_process = subprocess.Popen(
            worker_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        time.sleep(5)  # Give it time to initialize the model

        # Check if worker started successfully
        if worker_process.poll() is not None:
            stdout, stderr = worker_process.communicate()
            pytest.fail(f"Worker failed to start:\nstdout: {stdout}\nstderr: {stderr}")

        # 3. Start Spooler in a subprocess
        spooler_process = subprocess.Popen(
            [sys.executable, "-m", "cryoPARES.inference.daemon.spoolingFiller",
             "--directory", str(cryo_em_data),
             "--ip", "localhost", "--port", str(test_port), "--authkey", "test_key",
             "--pattern", "*.star", "--check_interval", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        # Give spooler time to find and queue the file
        time.sleep(3)

        # 4. Add poison pill to queue so worker exits after processing the file
        print("Adding poison pill to queue after input file...")
        try:
            from cryoPARES.inference.daemon.queueManager import connect_to_queue
            q, m = connect_to_queue(ip="localhost", port=test_port, authkey="test_key")
            q.put(None)
            print("✓ Poison pill added - worker should exit after processing")
        except Exception as e:
            print(f"Failed to add poison pill: {e}")

        # Monitor the system with timeout
        # The worker should process the file then exit when it hits the poison pill
        max_wait_time = 60  # 60 seconds should be plenty
        start_time = time.time()
        check_interval = 1  # Check more frequently

        print(f"\nWaiting for worker to process file and exit (max {max_wait_time}s)...")
        print("Monitoring processes and output files...")

        last_file_count = 0
        while time.time() - start_time < max_wait_time:
            time.sleep(check_interval)

            # Check process status
            queue_alive = queue_process.poll() is None
            worker_alive = worker_process.poll() is None
            spooler_alive = spooler_process.poll() is None

            # Check for any files in results directory
            result_files = list(results_dir.glob("*"))
            file_count = len(result_files)

            # Print status if files appeared or every 10 seconds
            elapsed = time.time() - start_time
            if file_count != last_file_count or (elapsed % 10 < check_interval):
                star_files = [f.name for f in result_files if f.name.endswith('.star')]
                print(f"  [{elapsed:.0f}s] Processes: Q={queue_alive}, W={worker_alive}, S={spooler_alive} | "
                      f"Files: {file_count} ({len(star_files)} .star)")
                if star_files:
                    print(f"    Star files: {star_files}")
                last_file_count = file_count

            # Check if worker has finished
            if not worker_alive:
                print(f"Worker process exited (code: {worker_process.returncode})")
                # Give a bit more time for files to be flushed
                time.sleep(2)
                break

            # Check for star files as completion indicator
            star_files = list(results_dir.glob("*.star"))
            if star_files:
                print(f"Found {len(star_files)} .star file(s), worker likely completed")
                # Wait a bit longer to ensure worker finishes cleanly
                time.sleep(3)
                break

        elapsed_total = time.time() - start_time
        print(f"\nMonitoring completed after {elapsed_total:.1f}s")
        print(f"Final process status: Q={queue_process.poll() is None}, "
              f"W={worker_process.poll() is None}, S={spooler_process.poll() is None}")

        # 5. Gracefully terminate remaining processes
        print("\nShutting down remaining daemon processes...")

        # Worker should have already exited from the poison pill
        if worker_process.poll() is None:
            print("⚠ Warning: Worker is still running (should have exited from poison pill)")
            print("  Sending another poison pill...")
            try:
                from cryoPARES.inference.daemon.queueManager import connect_to_queue
                q, m = connect_to_queue(ip="localhost", port=test_port, authkey="test_key")
                q.put(None)
            except Exception as e:
                print(f"  Could not send poison pill: {e}")
        else:
            print(f"✓ Worker exited normally (code: {worker_process.returncode})")

        # Terminate spooler
        if spooler_process and spooler_process.poll() is None:
            spooler_process.send_signal(signal.SIGINT)
            try:
                spooler_stdout, spooler_stderr = spooler_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                print("Spooler didn't terminate gracefully, killing...")
                spooler_process.kill()
                spooler_stdout, spooler_stderr = spooler_process.communicate()

            print("\n=== Spooler ===")
            if spooler_stdout:
                print("stdout:", spooler_stdout)
            if spooler_stderr:
                print("stderr:", spooler_stderr)

        # Worker should exit after receiving poison pill
        if worker_process and worker_process.poll() is None:
            try:
                worker_stdout, worker_stderr = worker_process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                print("Worker didn't terminate gracefully, killing...")
                worker_process.kill()
                worker_stdout, worker_stderr = worker_process.communicate()

            print("\n=== Worker ===")
            if worker_stdout:
                print("stdout:", worker_stdout)
            if worker_stderr:
                print("stderr:", worker_stderr)

        # Terminate queue manager
        if queue_process and queue_process.poll() is None:
            queue_process.send_signal(signal.SIGINT)
            try:
                queue_stdout, queue_stderr = queue_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                print("Queue manager didn't terminate gracefully, killing...")
                queue_process.kill()
                queue_stdout, queue_stderr = queue_process.communicate()

            print("\n=== Queue Manager ===")
            if queue_stdout:
                print("stdout:", queue_stdout)
            if queue_stderr:
                print("stderr:", queue_stderr)

        # 6. Verify the results
        print("\n=== Verifying Results ===")

        # List all files created for debugging
        all_files = list(results_dir.glob("*"))
        print(f"Files in results_dir: {[f.name for f in all_files]}")

        # Check for ANY star file with results (daemon uses different naming pattern)
        result_star_files = list(results_dir.glob("*.star"))
        print(f"Found {len(result_star_files)} result star files")

        assert len(result_star_files) > 0, f"No result star file was created. Files found: {[f.name for f in all_files]}"

        # Verify the star file is not empty
        for star_file in result_star_files:
            size = star_file.stat().st_size
            print(f"  - {star_file.name}: {size} bytes")
            assert size > 0, f"Star file {star_file.name} is empty"

        print(f"✓ Test passed - daemon successfully processed {len(result_star_files)} file(s)")

    finally:
        # Cleanup: make sure all processes are killed and port is freed
        print("\n=== Final Cleanup ===")
        killed_any = False
        for proc, name in [(queue_process, "queue"), (worker_process, "worker"), (spooler_process, "spooler")]:
            if proc and proc.poll() is None:
                print(f"Terminating {name} process (PID {proc.pid})...")
                proc.kill()
                proc.wait()
                killed_any = True

        if not killed_any:
            print("All processes already terminated cleanly")

        cleanup_port(test_port)
        print("✓ Cleanup complete")
        print("=== Test function exiting normally ===")
        # If you see "Killed" after this, it's from pytest's own cleanup, not this test
