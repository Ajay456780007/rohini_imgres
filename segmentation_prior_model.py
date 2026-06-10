import pydicom
from torch.autograd import Variable
import PIL
from PIL import Image
from deep_segmentation_prior.source.foreground_background._model import ConvAutoencoder, skip, collect_feature_maps
from deep_segmentation_prior.source.foreground_background._spectral import get_spectral_histogram, apply_lpf, apply_pca
from deep_segmentation_prior.source.foreground_background._cluster import get_cluster_segmentation
from deep_segmentation_prior.source.foreground_background._utils import *
from deep_segmentation_prior.source.foreground_background._edge_detection import cannyEdgeDetector
# from deep_segmentation_prior.source.foreground_background.generate_priors_and_hints import avg_recon_curve, model
import matplotlib
from Sub_Functions.Preprocessing import Preprocessing

matplotlib.use('TkAgg')

from deep_segmentation_prior.source.foreground_background.net import skip, get_noise

from matplotlib import pyplot as plt


def Segmentation_prior_model(pixel_array):
    # path = "Dataset/DB1/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"
    # ds = pydicom.dcmread(path)
    # pixel_array = ds.pixel_array
    img_float = pixel_array.astype(float)
    img_min = np.min(img_float)
    img_max = np.max(img_float)

    if img_max - img_min == 0:
        # Avoid zero-division errors if the image is completely blank
        normalized_img = np.zeros_like(img_float)
    else:
        normalized_img = (img_float - img_min) / (img_max - img_min)

    img_8bit = (normalized_img * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(img_8bit)
    pil_image = pil_image.resize((256, 256))
    # img_pil = Image.open(pil_image)
    img_pil = crop_image_by_multiplier(pil_image, d=32)

    img_np = pil_to_np(img_pil)

    # g = Preprocessing(img_np)
    # g.ROI_Extraction()

    device = 'cpu'

    gen_model = lambda: skip(
        2, 3,
        num_channels_down=[8, 16, 32],
        num_channels_up=[8, 16, 32],
        num_channels_skip=[0, 0, 0],
        upsample_mode='bilinear',
        filter_size_down=3,
        filter_size_up=3,
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

    learning_rate = 0.01
    mse_loss = nn.L1Loss()

    gen_optimizer = lambda: torch.optim.Adam(
        model.parameters(),
        lr=learning_rate)

    input_type = 'noise'
    input_depth = 2

    gen_noise = lambda: get_noise(
        input_depth,
        input_type,
        (img_np.shape[1], img_np.shape[2])) \
        .type(torch.FloatTensor) \
        .detach()

    num_iter = 20
    num_runs_for_stablility = 1

    reduce_ = lambda z: {
        k: np.median(np.array([d.get(k) for d in z]), axis=0)
        for k in set().union(*z)
    }

    X = torch.from_numpy(img_np).unsqueeze(0)
    X = Variable(X)

    total_recon_curve = []
    for run in range(num_runs_for_stablility):
        print(">>> run {} out of {} for stabliltiy".format(run + 1, num_runs_for_stablility))
        noise = gen_noise()
        model = gen_model()
        model.train()
        optimizer = gen_optimizer()

        learning_curve = []
        recon_curve = []

        for epoch in range(num_iter):

            X_rec = model(noise)  # inference

            loss = mse_loss(X_rec, X)
            learning_curve.append(loss)

            if (epoch + 1) % 10 == 0:
                print("epoch:: {}, LOSS = {}".format(epoch + 1, loss))
                X_rec_np = torch_to_np(X_rec)
                recon_curve.append((epoch + 1, X_rec_np))

            loss.backward()
            optimizer.step()
            model.zero_grad()

        total_recon_curve.append(dict(recon_curve))

    avg_recon_curve = reduce_(total_recon_curve)

    target_epochs = 20
    prior_result = avg_recon_curve[target_epochs]

    image = np_to_pil(prior_result)

    window_size = 25

    inputs = [img_np, prior_result]

    cluster_segmentations = []
    for image in inputs:
        result = get_cluster_segmentation(image, window_size=window_size, n_clusters=2)
        cluster_segmentations.append(result)

    # plot_image_grid([cluster_segmentations[-1]], factor=15)

    # plot_image_grid(np.transpose(cluster_segmentations[0],(1,2,0)), factor=15)
    segmented_one = np.transpose(cluster_segmentations[0], (1, 2, 0))
    # plt.imshow(l)
    return segmented_one
