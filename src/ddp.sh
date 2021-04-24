NUM_PROC=$1
shift
launcher=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
python $launcher --nnode=1 --node_rank=0 --nproc_per_node=$NUM_PROC main.py --local_world_size=$NUM_PROC
# python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC main.py "$@"