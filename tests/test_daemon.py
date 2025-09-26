
import os
import subprocess
import sys
import time
import pytest
from pathlib import Path
import signal
import torch
import yaml
from multiprocessing import Process

from cryoPARES.inference.daemon.queueManager import main as queue_main
from cryoPARES.inference.daemon.spoolingFiller import main as spooler_main
from cryoPARES.inference.daemon.daemonInference import DaemonInferencer
from cryoPARES.inference.daemon.materializePartialResults import materialize_partial_results

def run_queue_manager():
    queue_main(ip="localhost", port=50001, authkey="test_key")

def run_spooler(directory, pattern):
    spooler_main(directory=directory, ip="localhost", port=50001, authkey="test_key", pattern=pattern, check_interval=1)

def run_worker(checkpoint_dir, results_dir, reference_map):
    from cryoPARES.configManager.configParser import ConfigOverrideSystem
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.utils.paths import get_most_recent_file

    config_fname = get_most_recent_file(str(checkpoint_dir), "configs_*.yml")
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname)

    # We can't use the main function directly because of the argument parsing
    # and the config loading. We will instantiate the class directly.
    inferencer = DaemonInferencer(
        checkpoint_dir=str(checkpoint_dir),
        results_dir=str(results_dir),
        reference_map=str(reference_map),
        net_address="localhost",
        net_port=50001,
        net_authkey="test_key",
        model_halfset="half1",
        batch_size=2,
        num_data_workers=0,
        use_cuda=torch.cuda.is_available(),
        skip_localrefinement=False, # Must be False for reconstruction
        skip_reconstruction=False, # We want to test the reconstruction
        secs_between_partial_results_written=2
    )
    with inferencer:
        inferencer.run()


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
    return os.path.join(data_dir, "TEST")

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
                        "num_data_workers": 0,
                        "pin_memory": False,
                        "particlesDataset": {                "desired_sampling_rate_angs": 4.0,
                "desired_image_size_px": 64,
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


def test_daemon_mode(cryo_em_data, dummy_checkpoint, reference_map, tmp_path):
    results_dir = tmp_path / "daemon_results"
    results_dir.mkdir()

    # 1. Start Queue Manager in a separate process
    queue_process = Process(target=run_queue_manager)
    queue_process.start()
    time.sleep(2) # Give it time to start

    # 2. Start Worker in a separate process
    worker_process = Process(target=run_worker, args=(dummy_checkpoint, results_dir, reference_map))
    worker_process.start()
    time.sleep(5) # Give it time to initialize the model

    # 3. Start Spooler in a separate process
    spooler_process = Process(target=run_spooler, args=(cryo_em_data , "*.star"))
    spooler_process.start()

    # Let the system run for a bit
    # The spooler should find the files, the worker should process them,
    # and partial results should be written.
    time.sleep(15)

    # 4. Terminate the processes
    # Send poison pill to the queue to stop the worker gracefully
    from cryoPARES.inference.daemon.queueManager import connect_to_queue
    q, m = connect_to_queue(ip="localhost", port=50001, authkey="test_key")
    q.put(None)
    # The client-side manager does not have a shutdown method.
    # Rely on garbage collection to clean up the connection.
    
    # Terminate spooler and queue manager
    # We need to use SIGINT to trigger the graceful shutdown
    os.kill(spooler_process.pid, signal.SIGINT)
    spooler_process.join(timeout=5)
    if spooler_process.is_alive():
        os.kill(spooler_process.pid, signal.SIGKILL)

    worker_process.join(timeout=1000)
    if worker_process.is_alive():
        os.kill(worker_process.pid, signal.SIGKILL)

    os.kill(queue_process.pid, signal.SIGINT)
    queue_process.join(timeout=5)
    if queue_process.is_alive():
        os.kill(queue_process.pid, signal.SIGKILL)

    breakpoint()

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
