export GPUS="0," # change to your gpu id

for name in lan marc olek vlad
do
    python train_net.py --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS} silent True
    python run.py --type evaluate --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS}
done