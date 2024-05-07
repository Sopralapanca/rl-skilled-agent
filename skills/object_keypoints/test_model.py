import matplotlib.pyplot as plt
import numpy as np
from model import *
from dataset import Dataset, Sampler
from torchvision import transforms
from matplotlib.patches import Circle
from utils import get_n_colors
from PIL import Image, ImageDraw
import os
import torch



ENV = "BreakoutNoFrameskip-v4"
#model_path = "./saved_models/"+ENV+".pt"
model_path = ".././models/breakout-obj-key-expert.pt"

batch_size = 32
image_channels = 1
k = 4 # number of keypoints, the same as in train_model.py
#data_path = f"../../data/{ENV}"
data_path = f"../../data_expert/{ENV}"


encoder = Encoder(image_channels)
key_net = KeyNet(image_channels, k)
refine_net = RefineNet(image_channels)

model = Transporter(encoder, key_net, refine_net)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


transform = transforms.ToTensor()
dataset = Dataset(data_path, transform=transform)
sampler = Sampler(dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=sampler)

model.eval()
for xt, xtp1 in loader:
    break

print("xtp1", xtp1.shape) #torch.Size([16, 3, 84, 84])

target_keypoints = model.key_net(xtp1)
reconstruction = model(xt, xtp1)

out1 = model.encoder(xtp1) #usa questo
out2 = model.key_net(xtp1) #usa questo
print("Encoder", out1.shape)
print("keypoints", out2.shape) #number of images * number of keypoints founded * width * height
print("reconstruction", reconstruction.shape)

fig, ax = plt.subplots(1, 5, figsize=(12, 4))
idx = 12
k_idx = 1
std = 0.1

feature_maps = spatial_softmax(target_keypoints)
gmap = gaussian_map(feature_maps, std)[idx, k_idx]

ax[0].imshow(xt[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[1].imshow(xtp1[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[2].imshow(reconstruction[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[3].imshow(gmap.detach().numpy(), cmap='gray', vmin=0, vmax=1)
ax[4].imshow(feature_maps[idx, k_idx].detach().numpy(), cmap='gray')

ax[0].title.set_text('xt')
ax[1].title.set_text('xtp1+keypoints')
ax[2].title.set_text('Reconstruction')
ax[3].title.set_text('Gaussian Map')
ax[4].title.set_text('Feature Maps')

locs = compute_keypoint_location_mean(
    spatial_softmax(model.key_net(xtp1)))[idx]

print("locs", locs.shape)

# set keypoints over image
colors = get_n_colors(len(locs))
for i, l in enumerate((locs + 1) / 2 * 80):
    ax[1].add_patch(Circle((l[1].item(), l[0].item()), 2,
                           color=colors[i], alpha=0.5))


plt.show()

def get_keypoints(model, source_images):
    return compute_keypoint_location_mean(
        spatial_softmax(model.key_net(source_images)))


def annotate_keypoints(img, kp_t, colors):
    draw = ImageDraw.Draw(img)

    for i, kp in enumerate(unnormalize_kp(kp_t, image_width)):
        y = kp.detach().numpy()[0]
        x = kp.detach().numpy()[1]
        r = 2

        draw.ellipse((x - r, y - r, x + r, y + r), colors[i][0])

    return img


def unnormalize_kp(kp, img_width):
    return (kp + 1) / 2 * img_width


episode = 599
traj = torch.stack(dataset.get_trajectory(episode))
keypoints = get_keypoints(model, traj)
image_width = traj.size(-1)

fig = plt.figure(figsize=(12, 3))

colors = get_n_colors(k)
colors = [(int(color[0] * 255), int(color[1]*255), int(color[2]*255), 255) for color in colors]


trajectory_length = len([name for name in os.listdir(f"{data_path}/{episode}") if os.path.isfile(os.path.join(f"{data_path}/{episode}", name))])-1
steps = trajectory_length // 5
number_of_plots = (trajectory_length // steps) + 1
print("length", trajectory_length)
print("number of plots", number_of_plots)
for i, t in enumerate(range(0, trajectory_length, steps)):
    frame_t = traj[t].permute([1, 2, 0])
    kp_t = keypoints[t]
    frame_t = (frame_t.detach().numpy() * 255).astype('uint8')[:, :, 0]
    im = Image.fromarray(frame_t)
    annotate_keypoints(im, kp_t, colors)
    im = np.array(im)
    ax = fig.add_subplot(1, number_of_plots, i + 1)
    ax.imshow(im)
    ax.set_axis_off()
    ax.set_title(f't = {t}/{trajectory_length}')

plt.show()


traj = torch.stack(dataset.get_trajectory(episode))
keypoints = get_keypoints(model, traj)
image_width = traj.size(-1)

def get_heatmaps(model, source_images, normalize=True):
    if normalize:
        return spatial_softmax(model.key_net(source_images))
    return model.key_net(source_images)




fig, ax = plt.subplots(1, k+1, figsize=(12, 3))

idx = 16

ax[0].imshow(traj[idx].permute([1, 2, 0]).detach().numpy())
ax[0].set_axis_off()
ax[0].set_title(f't = {idx}')

hm = get_heatmaps(model, traj, normalize=False)[idx].detach().numpy()

for i, m in enumerate(hm):
    ax[i+1].imshow(m, cmap='gray')
    ax[i+1].set_axis_off()
    ax[i+1].set_title(f'keynet heatmap {i}')

plt.show()

fig, ax = plt.subplots(1, k+1, figsize=(12, 3))

idx = 16

ax[0].imshow(traj[idx].permute([1, 2, 0]).detach().numpy())
ax[0].set_axis_off()
ax[0].set_title(f't = {idx}')

hm = get_heatmaps(model, traj, normalize=True)[idx].detach().numpy()

for i, m in enumerate(hm):
    ax[i+1].imshow(m, cmap='gray')
    ax[i+1].set_axis_off()
    ax[i + 1].set_title(f'norm. keynet hm {i}')

plt.show()