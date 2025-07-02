import os
import time
from cryoPARES.projmatching.projMatching import  ProjectionMatcher
import torch.profiler


def profile_projection_matcher():
    """
    Profile the ProjectionMatcher with proper CUDA profiling setup.
    Sets necessary environment variables and uses torch.profiler for detailed analysis.
    """

    # Set environment variables for CUDA profiling
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA execution for accurate timing
    os.environ['TORCH_PROFILER_ENABLED'] = '1'  # Enable torch profiler
    os.environ['KINETO_LOG_LEVEL'] = '3'  # Reduce kineto logging verbosity

    print("Starting profiling with CUDA synchronization...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        # Warm up CUDA
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Setup profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def trace_handler(p):
        """Handle profiler output"""
        output_path = f"./projection_matcher_trace_{p.step_num}.json"
        p.export_chrome_trace(output_path)
        print(f"Profiler trace saved to: {output_path}")

        # Print top operations by CUDA time if available
        if device == "cuda":
            print("\nTop 10 operations by CUDA time:")
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print("\nTop 10 operations by CPU time:")
        print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Profile with context manager
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1,  # Wait 1 step before recording
                warmup=2,  # Warmup for 2 steps
                active=3,  # Record for 3 steps
                repeat=1  # Repeat cycle 1 time
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,  # Record tensor shapes
            profile_memory=True,  # Profile memory usage
            with_stack=True,  # Include stack traces
            with_flops=True  # Include FLOP counts
    ) as prof:

        # Create ProjectionMatcher instance
        pj = ProjectionMatcher(
            reference_vol="/home/sanchezg/tmp/cak_11799_usedTraining_ligErased.mrc",
            grid_distance_degs=12,
            grid_step_degs=2,
            pixel_size=None,
            filter_resolution_angst=5,
            max_shift_fraction=0.2,
            return_top_k=1
        )
        pj.to(device)

        # Create fake data for profiling
        batch_size = 4
        input_topk_mats = 1

        fakefimage = torch.rand(batch_size, *pj.reference_vol.shape[-2:],
                                dtype=torch.complex64, device=device)
        fakeCtf = torch.rand(batch_size, *pj.reference_vol.shape[-2:],
                             dtype=torch.float32, device=device)

        from scipy.spatial.transform import Rotation
        rotmats = torch.as_tensor(
            Rotation.random(batch_size, random_state=1).as_matrix(),
            dtype=torch.float32, device=device
        ).unsqueeze(1).repeat(1, input_topk_mats, 1, 1)

        print("Starting profiled execution...")

        # Run multiple steps for profiling
        for step in range(7):  # wait(1) + warmup(2) + active(3) + extra(1)
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure previous operations complete

            # Profile this operation
            start_time = time.time()
            maxCorrs, predRotMats, predShiftsAngs, comparedWeight = pj.align_particles(fakefimage, fakeCtf, rotmats)

            if device == "cuda":
                torch.cuda.synchronize()  # Wait for GPU operations to complete

            end_time = time.time()
            print(f"Step {step}: {end_time - start_time:.4f} seconds")

            prof.step()  # Signal profiler to move to next step

    # Memory profiling summary
    if device == "cuda":
        print(f"\nGPU Memory Summary:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    print("\nProfiler results saved. You can view the trace file in Chrome by:")
    print("1. Open Chrome and go to chrome://tracing")
    print("2. Load the generated .json trace file")
    print("3. Analyze the timeline and performance bottlenecks")


def profile_with_memory_tracking():
    """
    Alternative profiling function with detailed memory tracking
    """
    print("Starting memory-focused profiling...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Memory snapshot before
    if device == "cuda":
        memory_before = torch.cuda.memory_allocated()
        print(f"GPU Memory before: {memory_before / 1024 ** 2:.2f} MB")

    # Run the test
    start_time = time.time()
    _test0()  # or _test1()
    end_time = time.time()

    # Memory snapshot after
    if device == "cuda":
        memory_after = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"GPU Memory after: {memory_after / 1024 ** 2:.2f} MB")
        print(f"Peak GPU Memory: {peak_memory / 1024 ** 2:.2f} MB")
        print(f"Memory increase: {(memory_after - memory_before) / 1024 ** 2:.2f} MB")

    print(f"Total execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    # Original test
    # _test1()

    # Run profiling
    print("=" * 50)
    print("PROFILING MODE")
    print("=" * 50)

    # Choose profiling method
    try:
        profile_projection_matcher()
    except Exception as e:
        print(f"Detailed profiling failed: {e}")
        print("Falling back to simple memory tracking...")
        profile_with_memory_tracking()