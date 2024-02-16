import torch as th
from skill_models import *
from feature_extractors import CNNConcatExtractor, LinearConcatExtractor, CombineExtractor

device = "cuda:0"

s = []
s.append(get_state_rep_uns("pong", device))
s.append(get_object_keypoints_encoder("pong", device))
s.append(get_object_keypoints_keynet("pong", device))
s.append(get_video_object_segmentation("pong", device))

cnnextractor = CNNConcatExtractor(None, 256, s, 2, device)
print(cnnextractor)
linearextractor = LinearConcatExtractor(None, 256, s, device)
print(linearextractor)
combineextractor = CombineExtractor(None, 256, s, 1, device)
print(combineextractor)

inp = th.zeros(1, 4, 84, 84, device=device)
print(inp.shape)

out = cnnextractor(inp)
print("CNN:", out.shape)

out = linearextractor(inp)
print("Linear:", out.shape)

out = combineextractor(inp)
print("Combine:", out.shape)