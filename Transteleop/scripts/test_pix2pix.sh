set -ex

# training parameters
MODEL='pix2pix' # 'bicycle_gan'
NETG='resnet_3blocks' #
NETD='n_layers' # cureent default n_layersis is 2,  'basic' is 3. 'pixel'
NGF=32
NDF=32
NORM='batch'
NITER=100
NITER_DECAY=100

# visulization and save
CHECKPOINTS_DIR=./checkpoints/

# dataset
CLASS='../data/com_201809'
DIRECTION='AtoB'
LOAD_SIZE=96
CROP_SIZE=96
INPUT_NC=1
OUTPUT_NC=1
PREPROCESS='resize'

# naming
NAME='test'
GPU_ID=0

# command
python ./test.py \
  --gpu_ids ${GPU_ID} \
  ${STN} \
  --eval \
  --preprocess ${PREPROCESS} \
  --dataroot ${CLASS} \
  --dataset_mode 'joint' \
  --name ${NAME} \
  --model ${MODEL} \
  --netG ${NETG} \
  --netD ${NETD} \
  --fc_embedding \
  --g_embed 64 \
  --ndf ${NGF} \
  --ngf ${NDF} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --norm ${NORM} \
  --epoch '50'


