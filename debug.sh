#/!/bin/bash

python train_agent.py --env Pong --device 1 --use-skill True --debug True --extractor lin_concat_ext
echo "Linear Concat Finished"
python train_agent.py --env Pong --device 1 --use-skill True --debug True --extractor cnn_concat_ext
echo "CNN Concat Finished"
python train_agent.py --env Pong --device 1 --use-skill True --debug True --extractor combine_ext
echo "Combine Concat Finished"
python train_agent.py --env Pong --device 1 --use-skill True --debug True --extractor self_attention_ext
echo "SelfAttention Concat Finished"
python train_agent.py --env Pong --device 1 --use-skill True --debug True --extractor reservoir_concat_ext
echo "Reservoir Concat Finished"
