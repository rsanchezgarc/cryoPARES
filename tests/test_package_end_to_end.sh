#!/bin/bash
################################################################################
# End-to-End Test Script for CryoPARES
#
# This script tests the complete CryoPARES pipeline from installation to inference:
# 1. Creates a temporary conda environment
# 2. Installs cryoPARES from local repository
# 3. Downloads test data using cesped (cached in ~/tmp/cryoPARES_test_data)
# 4. Trains a small model
# 5. Runs inference on test particles
# 6. Verifies outputs and reports results
#
# Usage:
#   ./test_package_end_to_end.sh [--keep-env] [--keep-outputs] [--cpu-only]
#
# Options:
#   --keep-env       Don't delete the conda environment after test
#   --keep-outputs   Don't delete test outputs (training/inference results)
#   --cpu-only       Force CPU-only mode (skip GPU even if available)
#
################################################################################

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
KEEP_ENV=false
KEEP_OUTPUTS=false
CPU_ONLY=false
for arg in "$@"; do
    case $arg in
        --keep-env)
            KEEP_ENV=true
            ;;
        --keep-outputs)
            KEEP_OUTPUTS=true
            ;;
        --cpu-only)
            CPU_ONLY=true
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            exit 1
            ;;
    esac
done

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSISTENT_DATA_DIR="$HOME/tmp/cryoSupervisedDataset/"
TEST_WORK_DIR="/tmp/cryoPARES_test_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="cryopares_test_$(date +%Y%m%d_%H%M%S)"
N_PARTICLES=100
N_EPOCHS=3
START_TIME=$(date +%s)

# Function to print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

# Cleanup function
cleanup() {
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        print_error "Test failed with exit code $exit_code"
    fi

    # Deactivate conda environment if active
    if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
        print_step "Deactivating conda environment"
        conda deactivate || true
    fi

    # Remove conda environment unless --keep-env was specified
    if [ "$KEEP_ENV" = false ]; then
        print_step "Removing test conda environment: $ENV_NAME"
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    else
        print_info "Keeping conda environment: $ENV_NAME (use --keep-env to change)"
    fi

    # Remove test outputs unless --keep-outputs was specified
    if [ "$KEEP_OUTPUTS" = false ]; then
        if [ -d "$TEST_WORK_DIR" ]; then
            print_step "Removing test outputs: $TEST_WORK_DIR"
            rm -rf "$TEST_WORK_DIR"
        fi
    else
        print_info "Keeping test outputs: $TEST_WORK_DIR"
    fi

    # Calculate total time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ TEST PASSED${NC}"
    else
        echo -e "${RED}✗ TEST FAILED${NC}"
    fi
    echo "Total time: ${DURATION}s"
    echo "========================================"

    exit $exit_code
}

trap cleanup EXIT INT TERM

# Main test workflow
main() {
    echo "========================================"
    echo "CryoPARES End-to-End Test"
    echo "========================================"
    echo ""

    # Step 1: Check conda availability
    print_step "Checking conda availability"
    if ! command -v conda &> /dev/null; then
        print_error "conda not found. Please install conda first."
        exit 1
    fi
    print_info "conda found: $(conda --version)"

    # Step 2: Increase file descriptor limit
    print_step "Increasing file descriptor limit"
    ulimit -n 65536 || print_warning "Failed to increase ulimit (may cause issues with many .mrcs files)"
    print_info "Current ulimit: $(ulimit -n)"

    # Step 3: Create temporary conda environment
    print_step "Creating temporary conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.12 -y

    # Activate environment
    print_step "Activating conda environment"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Verify activation
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        print_error "Failed to activate conda environment"
        exit 1
    fi
    print_info "Active environment: $CONDA_DEFAULT_ENV"

    # Step 4: Install cryoPARES from local repository
    print_step "Installing cryoPARES from local repository"
    cd "$SCRIPT_DIR"
    pip install -e . --quiet
    print_info "cryoPARES installed"

    # Step 5: Install cesped for test data
    print_step "Installing cesped for test data download"
    pip install cesped --quiet
    print_info "cesped installed"

    # Step 6: Download/prepare test data
    print_step "Preparing test data (persistent location: $PERSISTENT_DATA_DIR)"
    mkdir -p "$PERSISTENT_DATA_DIR"

    # Create a small Python script to download data using cesped
    DOWNLOAD_SCRIPT="$TEST_WORK_DIR/download_data.py"
    mkdir -p "$TEST_WORK_DIR"

    cat > "$DOWNLOAD_SCRIPT" <<'PYTHON_EOF'
import os
import sys
import shutil
from pathlib import Path
import starfile
import pandas as pd

# Configuration
persistent_data_dir = sys.argv[1]
n_particles = int(sys.argv[2])
output_dir = sys.argv[3]

print(f"Persistent data directory: {persistent_data_dir}")
print(f"Output directory: {output_dir}")

# Download TEST dataset using cesped
os.makedirs(persistent_data_dir, exist_ok=True)
os.chdir(persistent_data_dir)

try:
    from cesped.particlesDataset import ParticlesDataset

    # Check if data already exists
    test_data_path = os.path.join(persistent_data_dir, "TEST")
    if not os.path.exists(test_data_path):
        print("Downloading TEST dataset with cesped (this may take a few minutes)...")
        ps = ParticlesDataset("TEST", halfset=1, benchmarkDir=persistent_data_dir)
        print("✓ Download complete")
    else:
        print("✓ Test data already exists, skipping download")

    # Find the star file
    star_files = list(Path(test_data_path).glob("*.star"))
    if not star_files:
        print("ERROR: No .star files found in test data")
        sys.exit(1)

    original_star = star_files[0]
    print(f"Using star file: {original_star}")

    # Create subset with limited particles
    os.makedirs(output_dir, exist_ok=True)
    data = starfile.read(original_star)

    # Handle both dict (multi-table) and dataframe (single table)
    particles_df = None
    if isinstance(data, dict):
        for key in data:
            if 'particles' in key.lower():
                data[key] = data[key].head(n_particles)
                particles_df = data[key]
                break
    else:
        data = data.head(n_particles)
        particles_df = data

    if particles_df is None:
        print("ERROR: Could not find particles table in star file")
        sys.exit(1)

    print(f"✓ Created subset with {len(particles_df)} particles")

    # Write subset star file
    subset_star = os.path.join(output_dir, "particles_subset.star")
    starfile.write(data, subset_star)
    print(f"✓ Wrote subset star file: {subset_star}")

    # Copy referenced mrcs files
    if 'rlnImageName' in particles_df.columns:
        mrcs_files = particles_df['rlnImageName'].str.split('@').str[1].unique()
        print(f"Copying {len(mrcs_files)} .mrcs files...")
        for mrcs in mrcs_files:
            src = Path(test_data_path) / mrcs
            dst = Path(output_dir) / mrcs
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        print("✓ .mrcs files copied")

    print(f"\n✓ Test data ready in: {output_dir}")

except ImportError as e:
    print(f"ERROR: Failed to import cesped: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to prepare test data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

    python "$DOWNLOAD_SCRIPT" "$PERSISTENT_DATA_DIR" "$N_PARTICLES" "$TEST_WORK_DIR/test_data"

    TEST_DATA_DIR="$TEST_WORK_DIR/test_data"
    TEST_STAR_FILE="$TEST_DATA_DIR/particles_subset.star"

    if [ ! -f "$TEST_STAR_FILE" ]; then
        print_error "Test star file not found: $TEST_STAR_FILE"
        exit 1
    fi

    # Step 7: Train a model WITH --NOT_split_halves
    print_step "Test 1: Training model with --NOT_split_halves (${N_EPOCHS} epochs, ${N_PARTICLES} particles)"
    TRAIN_OUTPUT_DIR="$TEST_WORK_DIR/training_output"

    TRAIN_CMD="python -m cryoPARES.train.train \
        --symmetry C1 \
        --particles_star_fname $TEST_STAR_FILE \
        --particles_dir $TEST_DATA_DIR \
        --train_save_dir $TRAIN_OUTPUT_DIR \
        --n_epochs $N_EPOCHS \
        --batch_size 8 \
        --num_dataworkers 0 \
        --NOT_split_halves \
        --config \
            train.learning_rate=1e-3 \
            models.image2sphere.lmax=6 \
            datamanager.particlesDataset.sampling_rate_angs_for_nnet=3.0 \
            datamanager.particlesDataset.image_size_px_for_nnet=64 \
            datamanager.num_augmented_copies_per_batch=1"

    if [ "$CPU_ONLY" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --NOT_use_cuda"
    fi

    print_info "Running: $TRAIN_CMD"
    eval "$TRAIN_CMD"

    # Verify training outputs
    CHECKPOINT_DIR="$TRAIN_OUTPUT_DIR/version_0"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "Training failed: checkpoint directory not found"
        exit 1
    fi
    print_info "✓ Training completed: $CHECKPOINT_DIR"

    # Step 8: Run inference on --NOT_split_halves model
    print_step "Test 1: Running inference on --NOT_split_halves model"
    INFERENCE_OUTPUT_DIR="$TEST_WORK_DIR/inference_output"

    INFER_CMD="python -m cryoPARES.inference.infer \
        --particles_star_fname $TEST_STAR_FILE \
        --checkpoint_dir $CHECKPOINT_DIR \
        --results_dir $INFERENCE_OUTPUT_DIR \
        --particles_dir $TEST_DATA_DIR \
        --batch_size 8 \
        --num_dataworkers 0 \
        --n_jobs 1"

    if [ "$CPU_ONLY" = true ]; then
        INFER_CMD="$INFER_CMD --NOT_use_cuda"
    fi

    print_info "Running: $INFER_CMD"
    eval "$INFER_CMD"

    # Verify inference outputs
    if [ ! -d "$INFERENCE_OUTPUT_DIR" ]; then
        print_error "Inference failed: output directory not found"
        exit 1
    fi

    # Check for expected output files
    EXPECTED_FILES=(
        "$INFERENCE_OUTPUT_DIR/particles_with_poses.star"
        "$INFERENCE_OUTPUT_DIR/map.mrc"
    )

    for file in "${EXPECTED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Expected file not found: $file"
        else
            print_info "✓ Found output: $(basename $file)"
        fi
    done

    print_step "Test 1 completed successfully!"

    # Step 9: Train a model WITHOUT --NOT_split_halves
    print_step "Test 2: Training model with split halves (${N_EPOCHS} epochs, ${N_PARTICLES} particles)"
    TRAIN_OUTPUT_DIR_SPLIT="$TEST_WORK_DIR/training_output_split"

    TRAIN_CMD_SPLIT="python -m cryoPARES.train.train \
        --symmetry C1 \
        --particles_star_fname $TEST_STAR_FILE \
        --particles_dir $TEST_DATA_DIR \
        --train_save_dir $TRAIN_OUTPUT_DIR_SPLIT \
        --n_epochs $N_EPOCHS \
        --batch_size 8 \
        --num_dataworkers 0 \
        --config \
            train.learning_rate=1e-3 \
            models.image2sphere.lmax=6 \
            datamanager.particlesDataset.sampling_rate_angs_for_nnet=3.0 \
            datamanager.particlesDataset.image_size_px_for_nnet=64 \
            datamanager.num_augmented_copies_per_batch=1"

    if [ "$CPU_ONLY" = true ]; then
        TRAIN_CMD_SPLIT="$TRAIN_CMD_SPLIT --NOT_use_cuda"
    fi

    print_info "Running: $TRAIN_CMD_SPLIT"
    eval "$TRAIN_CMD_SPLIT"

    # Verify training outputs
    CHECKPOINT_DIR_SPLIT="$TRAIN_OUTPUT_DIR_SPLIT/version_0"
    if [ ! -d "$CHECKPOINT_DIR_SPLIT" ]; then
        print_error "Training failed: checkpoint directory not found"
        exit 1
    fi
    print_info "✓ Training completed: $CHECKPOINT_DIR_SPLIT"

    # Step 10: Run inference on split halves model
    print_step "Test 2: Running inference on split halves model"
    INFERENCE_OUTPUT_DIR_SPLIT="$TEST_WORK_DIR/inference_output_split"

    INFER_CMD_SPLIT="python -m cryoPARES.inference.infer \
        --particles_star_fname $TEST_STAR_FILE \
        --checkpoint_dir $CHECKPOINT_DIR_SPLIT \
        --results_dir $INFERENCE_OUTPUT_DIR_SPLIT \
        --particles_dir $TEST_DATA_DIR \
        --batch_size 8 \
        --num_dataworkers 0 \
        --n_jobs 1"

    if [ "$CPU_ONLY" = true ]; then
        INFER_CMD_SPLIT="$INFER_CMD_SPLIT --NOT_use_cuda"
    fi

    print_info "Running: $INFER_CMD_SPLIT"
    eval "$INFER_CMD_SPLIT"

    # Verify inference outputs
    if [ ! -d "$INFERENCE_OUTPUT_DIR_SPLIT" ]; then
        print_error "Inference failed: output directory not found"
        exit 1
    fi

    # Check for expected output files
    EXPECTED_FILES_SPLIT=(
        "$INFERENCE_OUTPUT_DIR_SPLIT/particles_with_poses.star"
        "$INFERENCE_OUTPUT_DIR_SPLIT/map.mrc"
    )

    for file in "${EXPECTED_FILES_SPLIT[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Expected file not found: $file"
        else
            print_info "✓ Found output: $(basename $file)"
        fi
    done

    print_step "Test 2 completed successfully!"

    print_step "All tests completed successfully!"

    # Summary
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Persistent data dir: $PERSISTENT_DATA_DIR"
    echo "Test work dir:       $TEST_WORK_DIR"
    echo "Checkpoint dir:      $CHECKPOINT_DIR"
    echo "Inference output:    $INFERENCE_OUTPUT_DIR"
    echo "Conda environment:   $ENV_NAME"
    echo ""

    if [ "$KEEP_ENV" = true ]; then
        echo "To use the test environment:"
        echo "  conda activate $ENV_NAME"
    fi

    if [ "$KEEP_OUTPUTS" = true ]; then
        echo "To inspect outputs:"
        echo "  ls -lh $TEST_WORK_DIR"
    fi
}

# Run main workflow
main
