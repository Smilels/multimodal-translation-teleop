set -ex

# training parameters
MODEL='aejoint'
NETG='jointpost' # context'
NGF=32
NORM='batch'
BS=32
LR=0.002
NITER=50
NITER_DECAY=100

# visulization and save
DISPLAY_ID=11
PORT=8097
CHECKPOINTS_DIR=./checkpoints/
SAVE_EPOCH=20
STN='' #'--stn'
L1=1
Lambda_Joint=10
KEY=-1
EMBED=128
L2loss=''

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
NAME_BASE=${DATE}_${MODEL}_${NETG}${NGF}_bs${BS}lr${LR}ep${N_EPOCH}l1${L1}lj${Lambda_Joint}key${KEY}g${EMBED}_${PREPROCESS}_${STN}_${L2loss}
NAME=''
DISPLAY_ENV=${NAME_BASE}_${NAME}
GPU_ID=1

# command
python ./train_joint.py \
  --gpu_ids ${GPU_ID} \
  --display_id ${DISPLAY_ID} \
  --lambda_L1 ${L1} \
  ${STN} \
  --preprocess ${PREPROCESS} \
  --dataroot ${CLASS} \
  --name ${NAME_BASE}_${NAME} \
  --batch_size ${BS} \
  --model ${MODEL} \
  --netG ${NETG} \
  --fc_embedding \
  --g_embed ${EMBED} \
  --ndf ${NGF} \
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
  --lambda_Joint ${Lambda_Joint} \
  --dataset_mode 'joint' \
  --lr_policy 'step' \
  --lr_decay_iters 30 \
  ${L2loss}


