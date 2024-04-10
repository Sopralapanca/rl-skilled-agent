#/!/bin/bash


python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor lin_concat_ext --pi 512 256 --vf 512 256
echo "Linear Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor fixed_lin_concat_ext --pi 512 256 --vf 512 256
echo "Fixed Linear Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor cnn_concat_ext --pi 512 256 --vf 512 256
echo "CNN Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor combine_ext --pi 512 256 --vf 512 256
echo "Combine Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor self_attention_ext --pi 512 256 --vf 512 256
echo "Self Attention Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor dotproduct_attention_ext --pi 512 256 --vf 512 256
echo "Dot Product Attention Concat Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug True --extractor reservoir_concat_ext --pi 512 256 --vf 512 256
echo "Reservoir Concat Finished"
