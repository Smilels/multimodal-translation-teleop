set -ex

# training parameters
MODEL='aejoint' # 'bicycle_gan'
NETG='human' # 'hccnn_2'
NGF=32
NORM='jointpost'
NITER=100
NITER_DECAY=100
EMBED=128

# visulization and save
CHECKPOINTS_DIR=./checkpoints/

# dataset
DIRECTION='AtoB'
LOAD_SIZE=96
CROP_SIZE=96
INPUT_NC=1
OUTPUT_NC=1
STN=''

# naming
NAME='ae4down96'
GPU_ID=-1

# command
python ./shadow_demo_moveit.py \
  --gpu_ids ${GPU_ID} \
  --dataroot '' \
  ${STN} \
  --name ${NAME} \
  --model ${MODEL} \
  --netG ${NETG} \
  --fc_embedding \
  --g_embed  ${EMBED} \
  --ngf ${NGF} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --norm ${NORM} \
  --epoch 'latest' \
  --eval \
  --demo
