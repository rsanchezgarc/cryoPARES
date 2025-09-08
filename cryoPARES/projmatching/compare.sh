REF_VOL=~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc
STAR_F=~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star
PARTS_DIR=~/cryo/data/preAlignedParticles/EMPIAR-10166/data
SYM="C1"

#SIMULATION_NAME="6ACHSimulation"
#REF_VOL=~/cryo/data/preAlignedParticles/${SIMULATION_NAME}/reconstruction/relion_reconstruct_half1.mrc
#STAR_F=~/cryo/data/preAlignedParticles/${SIMULATION_NAME}/particleImgs/particles.star
#PARTS_DIR=~/cryo/data/preAlignedParticles/${SIMULATION_NAME}/particleImgs/
#SYM="D4"

BATCH_SIZE=2
GRID_DISTANCE=10
GRID_STEP=5
LIMIT_N_PARTS=100
OLD_RESULT_FNAME=/tmp/cmdPruebaAlign.star
NEW_RESULT_FNAME=/tmp/cmdPruebaProjMat.star

cd ~/cryo/myProjects/torchCryoAlign/ || exit
rm $OLD_RESULT_FNAME
/home/sanchezg/app/anaconda3/envs/torchCryoAlign/bin/python -m torchCryoAlign.alignerFourier \
      --reference_vol $REF_VOL  --star_in_fname $STAR_F --particles_root_dir $PARTS_DIR \
      --padding_factor 0.  --filter_resolution_angst 6\
      --batch_size $BATCH_SIZE --grid_resolution_degs $GRID_STEP --grid_distance_degs $GRID_DISTANCE \
      --limit_to_n_particles $LIMIT_N_PARTS  --star_out_fname $OLD_RESULT_FNAME

cd ~/cryo/myProjects/cryoPARES/ || exit
/home/sanchezg/app/anaconda3/envs/ddCryoEM2/bin/python -m cryoPARES.scripts.compare_poses $STAR_F $OLD_RESULT_FNAME --sym $SYM

echo "############################"
echo "CRYOPARES"
/home/sanchezg/app/anaconda3/envs/ddCryoEM2/bin/python -m cryoPARES.projmatching.projMatching \
    --reference_vol $REF_VOL --star_fname $STAR_F --particles_dir $PARTS_DIR \
    --batch_size $BATCH_SIZE --grid_step_degs $GRID_STEP --grid_distance_degs $((2 * $GRID_DISTANCE)) \
    --n_first_particles $LIMIT_N_PARTS --show_debug_stats --out_fname $NEW_RESULT_FNAME
/home/sanchezg/app/anaconda3/envs/ddCryoEM2/bin/python -m cryoPARES.scripts.compare_poses $STAR_F $NEW_RESULT_FNAME --sym $SYM # --align-frames --plot-all

