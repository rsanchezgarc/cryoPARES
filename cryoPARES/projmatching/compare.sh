REF_VOL=~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc
STAR_F=~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star
PARTS_DIR=~/cryo/data/preAlignedParticles/EMPIAR-10166/data
BATCH_SIZE=2
GRID_DISTANCE=8
GRID_STEP=2
LIMIT_N_PARTS=64
OLD_RESULT_FNAME=/tmp/cmdPruebaAlign.star
NEW_RESULT_FNAME=/tmp/cmdPruebaProjMat.star

cd ~/cryo/myProjects/torchCryoAlign/ || exit
rm $OLD_RESULT_FNAME
/home/sanchezg/app/anaconda3/envs/torchCryoAlign/bin/python -m torchCryoAlign.alignerFourier \
      --reference_vol $REF_VOL  --star_in_fname $STAR_F --particles_root_dir $PARTS_DIR \
      --batch_size $BATCH_SIZE --grid_resolution_degs $GRID_STEP --grid_distance_degs $GRID_DISTANCE \
      --limit_to_n_particles $LIMIT_N_PARTS  --star_out_fname $OLD_RESULT_FNAME

echo "CRYOPARES"
cd ~/cryo/myProjects/cryoPARES/ || exit
/home/sanchezg/app/anaconda3/envs/ddCryoEM2/bin/python -m cryoPARES.projmatching.projMatching \
    --reference_vol $REF_VOL --star_fname $STAR_F --particles_dir $PARTS_DIR \
    --batch_size $BATCH_SIZE --grid_step_degs $GRID_STEP --grid_distance_degs $((2 * $GRID_DISTANCE)) \
    --n_first_particles $LIMIT_N_PARTS --show_debug_stats --out_fname $NEW_RESULT_FNAME

