set -ex

# training parameters
MODEL='aejoint' # 'bicycle_gan'
NETG='jointpost' # 'hccnn_2'
NGF=32
NORM='batch'
NITER=100
NITER_DECAY=100
STN='' #'--stn'

# visulization and save
CHECKPOINTS_DIR=./checkpoints/

# dataset
CLASS='../data/com_201809'
DIRECTION='AtoB'
LOAD_SIZE=96
CROP_SIZE=96
INPUT_NC=1
OUTPUT_NC=1
PREPROCESS='resize' #resize_and_crop None

# naming
NAME='test'
GPU_ID=1

# command
python ./test_joint.py \
  --gpu_ids ${GPU_ID} \
  ${STN} \
  --preprocess ${PREPROCESS} \
  --dataroot ${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --netG ${NETG} \
  --fc_embedding \
  --g_embed 128 \
  --ngf ${NGF} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --norm ${NORM} \
  --epoch 'latest' \
  --dataset_mode 'joint' \
  --eval \
  --demo