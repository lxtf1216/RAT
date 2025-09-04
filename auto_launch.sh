#!/bin/bash
# 用法: ./auto_launch.sh <需要的GPU数量> <训练脚本> [脚本参数...]
# 示例: ./auto_launch.sh 2 test_ddp_min.py --arg1 val1

set -e

NUM_GPUS=$1
shift
SCRIPT=$1
shift
ARGS=$@

# 1. 检测空闲 GPU
# 阈值显存 2000 MB 以下视为空闲
FREE_GPUS=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits \
    | awk -F', ' '$1 < 2000 {print $2}' \
    | head -n $NUM_GPUS | xargs | tr ' ' ',')

if [ -z "$FREE_GPUS" ]; then
  echo "[ERROR] 没有找到足够空闲的 GPU (需要 $NUM_GPUS 张) "
  exit 1
fi

echo "[INFO] 选择 GPU: $FREE_GPUS"

# 2. 设置分布式相关环境变量
export CUDA_VISIBLE_DEVICES=$FREE_GPUS
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS

# 自动选一个网卡 (取第一个非lo的接口)，你可以手动改成 eno1/eth0
IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
export NCCL_SOCKET_IFNAME=$IFACE
echo "[INFO] 使用网络接口: $IFACE"

# 3. 启动 torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS $SCRIPT $ARGS
