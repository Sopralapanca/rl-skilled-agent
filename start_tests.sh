#/!/bin/bash

python train_agent.py --env Pong --device 2 --use-skill False --debug False
echo "Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug False --extractor lin_concat_ext
echo "Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug False --extractor cnn_concat_ext
echo "Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug False --extractor combine_ext
echo "Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug False --extractor self_attention_ext
echo "Finished"
python train_agent.py --env Pong --device 2 --use-skill True --debug False --extractor reservoir_ext
echo "Finished"

