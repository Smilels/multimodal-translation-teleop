set -ex

# training parameters
MODEL='pix2pix' # 'bicycle_gan'
NETG='resnet_3blocks' # '
NETD='n_layers' # cureent default n_layersis is 2
COND='' # '--conditional_D'
NGF=32
NDF=32
NORM='batch'
BS=16
LR=0.0002
NITER=50
NITER_DECAY=100

# visulization and save
DISPLAY_ID=11
PORT=8097
CHECKPOINTS_DIR=./checkpoints/
SAVE_EPOCH=10
STN='' #'--stn'
L1=150
Lambda_GAN=1
Lambda_D=0.1
EMBED=4000

# dataset
CLASS='../data/com_201809'
DIRECTION='AtoB'
LOAD_SIZE=96
CROP_SIZE=96
INPUT_NC=1
OUTPUT_NC=1
PREPROCESS='resize_jitter' #resize_and_crop None

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=${NITER}+${NITER_DECAY}
NAME_BASE=${DATE}_${MODEL}_${NGF}_${NDF}_${NETG}${NETD}_bs${BS}lr${LR}ep${N_EPOCH}l1${L1}lambda_g${Lambda_GAN}lambda_d${Lambda_D}g${EMBED}_${PREPROCESS}_${STN}${COND}
NAME='olddata'
DISPLAY_ENV=${NAME_BASE}_${NAME}
GPU_ID=3

# command
python ./train.py \
  --gpu_ids ${GPU_ID} \
  --display_id ${DISPLAY_ID} \
  --lambda_L1 ${L1} \
  --lambda_D ${Lambda_D} \
  --lambda_GAN ${Lambda_GAN} \
  ${STN} \
  ${COND} \
  --preprocess ${PREPROCESS} \
  --dataroot ${CLASS} \
  --name ${NAME_BASE}_${NAME} \
  --batch_size ${BS} \
  --model ${MODEL} \
  --netG ${NETG} \
  --netD ${NETD} \
  --fc_embedding \
  --g_embed ${EMBED} \
  --ndf ${NGF} \
  --ngf ${NDF} \
  --display_port ${PORT} \
  --display_env ${DISPLAY_ENV} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --norm ${NORM} \
  --pool_size 50 \
  --lr ${LR} \
  --n_epochs ${NITER} \
  --n_epochs_decay ${NITER_DECAY} \


