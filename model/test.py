""" This script requires TFC v2 ('pip install tensorflow-compression==2.*'). """
""" Parts of this are based on the code for the Minnen and Singh's "Channel-wise Autoregressive Entropy Models for Learned Image Compression" paper."""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import functools
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import glob
import sys
import tensorflow_datasets as tfds
from absl import app
from absl.flags import argparse_flags
import time
import numpy as np

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)

def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3

def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize,3))
  return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)

def dataset_processing(init_dataset,patch_size):
  data_set = []
  for img in init_dataset:
    image = read_png(img)
    check = check_image_size(image, patch_size)
    if check == True:
      data_set.append(img)
  return data_set

def dataset_split(proc_dataset):
  ds_D = []
  original_ds = []

  for img in proc_dataset:
    image1 = read_png(img)
    image = crop_image(image1, 256)
    original_ds.append(image)
    datum = image.numpy()
    data0 = datum[:,:,0]
    data1 = datum[:,:,1]
    data2 = datum[:,:,2]
    ds_R = tf.stack([data0, data1, data2], axis=2)
    ds_D.append(ds_R)

  return ds_D, original_ds

def get_custom_dataset(split, train_path):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files_init = glob.glob(train_path)
    if not files_init:
      raise RuntimeError(f"No training images found with glob "
                         f"'{train_path}'.")

    files1 = dataset_processing(files_init, 256)
    dsD, original_ds = dataset_split(files1)
    dsD1 = tf.data.Dataset.from_tensor_slices(dsD)
    #dsD1 = dsD1.shuffle(len(dsD), reshuffle_each_iteration=True)
    # if split == "train":
      # dsD1 = dsD1.repeat()
    dsD1 = dsD1.batch(1, drop_remainder=True)

  return dsD1, original_ds

class MaskedConv2D(tf.keras.layers.Layer):
    """PixelCNN-style masked convolution with dynamic input channel support."""

    def __init__(self, filters, kernel_size, mask_type='A'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.conv = None
        self.last_in_channels = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if self.conv is None or self.last_in_channels != in_channels:
            self.conv = tf.keras.layers.Conv2D(
                self.filters, self.kernel_size, padding="same", use_bias=True)
            self.conv.build(input_shape)

            # Build mask as a non-trainable variable (tracing safe)
            import numpy as np
            kh, kw = self.kernel_size
            mask = np.ones((kh, kw, in_channels, self.filters), dtype=np.float32)

            center_h, center_w = kh // 2, kw // 2
            mask[center_h, center_w + (1 if self.mask_type == 'B' else 0):, :, :] = 0
            mask[center_h + 1:, :, :, :] = 0

            self.mask = self.add_weight(
                name="mask",
                shape=mask.shape,
                initializer=tf.constant_initializer(mask),
                trainable=False,
            )
            self.last_in_channels = in_channels

    def call(self, x):
        # Ensure the conv is built if not yet done
        if self.conv is None or self.last_in_channels != x.shape[-1]:
            self.build(x.shape)

        masked_kernel = self.conv.kernel * self.mask
        outputs = tf.nn.conv2d(
            x, masked_kernel,
            strides=self.conv.strides,
            padding=self.conv.padding.upper(),
            data_format='NHWC'
        )
        if self.conv.use_bias:
            outputs = tf.nn.bias_add(outputs, self.conv.bias, data_format='NHWC')
        return outputs

class MaskedPixelCNNContext(tf.keras.layers.Layer):
    def __init__(self, latent_depth, num_slices, num_filters=128):
        super().__init__()
        self.num_slices = num_slices
        self.latent_depth = latent_depth
        self.slice_depth = latent_depth // num_slices
        self.max_in_channels = latent_depth  # e.g., 320 if latent space is 320

        # 1x1 projection to unify input channels to latent_depth
        self.input_projection = tf.keras.layers.Conv2D(
            self.latent_depth,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="context_input_projection"
        )

        # Masked convolutions
        self.masked_conv1 = MaskedConv2D(num_filters, (5, 5), mask_type='A')
        self.masked_conv2 = MaskedConv2D(num_filters, (5, 5), mask_type='B')
        self.masked_conv3 = MaskedConv2D(self.slice_depth, (3, 3), mask_type='B')

        # Optional activations
        self.act1 = tf.keras.layers.Conv2D(num_filters, 1, padding="same", activation='sigmoid')
        self.act2 = tf.keras.layers.Conv2D(num_filters, 1, padding="same", activation='sigmoid')

    def call(self, y_hat_slices):
        # Concatenate along channels
        y_hat = tf.concat(y_hat_slices, axis=-1)

        # Try to get static channel count
        in_channels = y_hat.shape[-1]
        if in_channels is None:
            in_channels = tf.shape(y_hat)[-1]

        pad_channels = self.max_in_channels - in_channels

        # Always apply dynamic padding (safe for gradient flow)
        padding = tf.stack([[0, 0], [0, 0], [0, 0], [0, self.max_in_channels - tf.shape(y_hat)[-1]]])
        y_hat = tf.pad(y_hat, padding)
        y_hat.set_shape([None, None, None, self.max_in_channels])

        # Enforce static shape for Conv2D build
        y_hat.set_shape([None, None, None, self.max_in_channels])

        # Apply projection to standard latent_depth
        y_hat = self.input_projection(y_hat)

        # Run through masked conv layers
        context = self.masked_conv1(y_hat)
        context = context * self.act1(context)
        context = self.masked_conv2(context)
        context = context * self.act2(context)
        context = self.masked_conv3(context)

        return context

class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, latent_depth):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        tf.keras.layers.Lambda(lambda x: x / 255.),
        conv(192, (5, 5), name="layer_0", activation=tfc.GDN(name="gdn_0")),
        conv(192, (5, 5), name="layer_1", activation=tfc.GDN(name="gdn_1")),
        conv(192, (5, 5), name="layer_2", activation=tfc.GDN(name="gdn_2")),
        conv(latent_depth, (5, 5), name="layer_3", activation=None),
    ]
    for layer in layers:
      self.add(layer)
    self.build(input_shape=(None, None, None, 3))

class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        conv(192, (5, 5), name="layer_0",
             activation=tfc.GDN(name="igdn_0", inverse=True)),
        conv(192, (5, 5), name="layer_1",
             activation=tfc.GDN(name="igdn_1", inverse=True)),
        conv(192, (5, 5), name="layer_2",
             activation=tfc.GDN(name="igdn_2", inverse=True)),
        conv(3, (5, 5), name="layer_3",
             activation=None),
        tf.keras.layers.Lambda(lambda x: x * 255.),
    ]
    for layer in layers:
      self.add(layer)

class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, hyperprior_depth):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, padding="same_zeros")

    # See Appendix C.2 for more information on using a small hyperprior.
    layers = [
        conv(320, (3, 3), name="layer_0", strides_down=1, use_bias=True,
             activation=tf.nn.relu),
        conv(256, (5, 5), name="layer_1", strides_down=2, use_bias=True,
             activation=tf.nn.relu),
        conv(hyperprior_depth, (5, 5), name="layer_2", strides_down=2,
             use_bias=False, activation=None),
    ]
    for layer in layers:
      self.add(layer)

class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(
        tfc.SignalConv2D, corr=False, padding="same_zeros", use_bias=True,
        kernel_parameter="variable", activation=tf.nn.relu)

    self.add(conv(192, (5, 5), name="layer_0", strides_up=2))
    self.add(MaskedConv2D(192, (3, 3), mask_type='B'))  # Contextual modelling
    self.add(conv(256, (5, 5), name="layer_1", strides_up=2))
    self.add(MaskedConv2D(256, (3, 3), mask_type='B'))  # Contextual modelling
    self.add(conv(320, (3, 3), name="layer_2", strides_up=1))

class SliceTransform(tf.keras.layers.Layer):
    def __init__(self, num_filters=192, reduction_ratio=16):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")

        # Squeeze-and-Excitation block
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(num_filters // reduction_ratio, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_filters, activation="sigmoid")
        self.reshape = tf.keras.layers.Reshape((1, 1, num_filters))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Apply SE attention
        se = self.global_avg_pool(x)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        x = x * se  # Channel-wise scaling

        return x

class ARCHEModel(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda,
               num_filters, latent_depth, hyperprior_depth,
               num_slices, max_support_slices,
               num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    self.num_slices = num_slices
    self.max_support_slices = max_support_slices
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.context_model = MaskedPixelCNNContext(latent_depth, self.num_slices)
    self.analysis_transform = AnalysisTransform(latent_depth)
    self.synthesis_transform = SynthesisTransform()
    self.hyper_analysis_transform = HyperAnalysisTransform(hyperprior_depth)
    self.hyper_synthesis_mean_transform = HyperSynthesisTransform()
    self.hyper_synthesis_scale_transform = HyperSynthesisTransform()
    self.cc_mean_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.cc_scale_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.lrp_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.lrp_scale_factors = [
        self.add_weight(
            name=f'lrp_scale_{i}',
            shape=(),
            initializer=tf.constant_initializer(0.6),
            trainable=True
        ) for i in range(self.num_slices)
    ]
    self.lrp_norms = [
        tf.keras.layers.LayerNormalization(axis=-1, name=f'lrp_norm_{i}')
        for i in range(self.num_slices)
    ]
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
    self.build((None, None, None, 3))
    # The call signature of decompress() depends on the number of slices, so we
    # need to compile the function dynamically.

  def call(self, x, training):
    """Computes rate and distortion losses."""
    x = tf.cast(x, self.compute_dtype)  # TODO(jonarchist): Why is this necessary?
    # Build the encoder (analysis) half of the hierarchical autoencoder.
    y = self.analysis_transform(x)
    y_shape = tf.shape(y)[1:-1]

    z = self.hyper_analysis_transform(y)

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[1:-1]), tf.float32)

    # Build the entropy model for the hyperprior (z).
    em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False,
        offset_heuristic=False)

    # When training, z_bpp is based on the noisy version of z (z_tilde).
    _, z_bits = em_z(z, training=training)
    z_bpp = tf.reduce_mean(z_bits) / num_pixels

    # Use rounding (instead of uniform noise) to modify z before passing it
    # to the hyper-synthesis transforms. Note that quantize() overrides the
    # gradient to create a straight-through estimator.
    z_hat = em_z.quantize(z)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)


    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = tf.split(y, self.num_slices, axis=-1)
    y_hat_slices = []
    y_bpps = []


    for slice_index, y_slice in enumerate(y_slices):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]


      # Apply Masked PixelCNN-based Context Model with reduced weight
      context = self.context_model(y_hat_slices if y_hat_slices else [tf.zeros_like(y_slice)])

      # Extract only the slice's part of sigma
      slice_depth = context.shape[-1]  # or self.latent_depth // self.num_slices
      start = slice_index * slice_depth
      end = (slice_index + 1) * slice_depth
      mu_slice = mu[:, :, :, start:end]
      sigma_slice = sigma[:, :, :, start:end]
      sigma_slice += 0.5 * context

      # Build a conditional entropy model for the slices.
      em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=3, compression=False)


      _, slice_bits = em_y(y_slice, sigma_slice, loc=mu_slice, training=training)
      slice_bpp = tf.reduce_mean(slice_bits) / num_pixels
      y_bpps.append(slice_bpp)

      # For the synthesis transform, use rounding. Note that quantize()
      # overrides the gradient to create a straight-through estimator.
      y_hat_slice = y_slice + tf.stop_gradient(tf.round(y_slice) - y_slice)  # ST Estimator / Smooth To Hard Quantization

      # Add latent residual prediction (LRP).
      # LRP support: mean + y_hat_slice
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)

      # Pass through LRP transform network
      lrp = self.lrp_transforms[slice_index](lrp_support)

      # 1. Apply normalization (LayerNorm across channels)
      lrp = self.lrp_norms[slice_index](lrp)

      # 2. Apply softsign activation for stability
      lrp = tf.nn.softsign(lrp)

      # 3. Apply trainable scale (reuse or define once in __init__)
      lrp = self.lrp_scale_factors[slice_index] * lrp
      lrp = lrp[:, :, :, start:end]

      # 4. Residual add
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the image reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)

    # Total bpp is sum of bpp from hyperprior and all slices.
    total_bpp = tf.add_n(y_bpps + [z_bpp])

    # Mean squared error across pixels.
    # Don't clip or round pixel values while training.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, total_bpp.dtype)


    psnr = tf.image.psnr(x, x_hat, max_val=255.)
    psnr = tf.cast(psnr, total_bpp.dtype)
    ssim = tf.image.ssim(x, x_hat, max_val=255.)
    ssim = tf.cast(ssim, total_bpp.dtype)
    ms_ssim = tf.image.ssim_multiscale(x, x_hat, max_val=255.)
    ms_ssim = tf.cast(ms_ssim, total_bpp.dtype)

    # Calculate and return the rate-distortion loss: R + lambda * D.
    loss = total_bpp + self.lmbda * mse

    if training:
      return loss, total_bpp, mse, psnr, ssim, ms_ssim
    else:
      return loss, total_bpp, mse, psnr, ssim, ms_ssim, x_hat

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse, psnr, ssim, ms_ssim = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.psnr.update_state(psnr)
    self.ssim.update_state(ssim)
    self.ms_ssim.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr, self.ssim, self.ms_ssim]}

  def test_step(self, x):
    loss, bpp, mse, psnr, ssim, ms_ssim, x_hat = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.psnr.update_state(psnr)
    self.ssim.update_state(ssim)
    self.ms_ssim.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr, self.ssim, self.ms_ssim, x_hat]}

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    self.psnr = tf.keras.metrics.Mean(name="psnr")
    self.ssim = tf.keras.metrics.Mean(name="ssim")
    self.ms_ssim = tf.keras.metrics.Mean(name="ms_ssim")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True,
        offset_heuristic=False)
    self.em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=3, compression=True)
    return retval

testdataset, original_ds = get_custom_dataset("test", "Give filepath")

model = ARCHEModel(
    lmbda = 0.01, num_filters = 192, latent_depth = 320, hyperprior_depth = 192,
    num_slices = 10, max_support_slices = 9, num_scales = 64, scale_min = 0.11,
    scale_max = 256.)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

model.summary()

dummy = np.random.rand(1,256,256,3).astype(np.float32)
_ = model.predict(dummy)
model.load_weights('Give filepath',by_name=True)
print("\n--- Starting model inference timing ---")
start_time = time.time()
test_results = model.predict(testdataset)
try:
    from tensorflow.python.eager import context
    context.async_wait()  # Works across TF 2.x versions
except Exception:
    pass  # Safe no-op if running on CPU

total_time = time.time() - start_time

# Estimate total number of images
try:
    num_images = len(list(testdataset))
except TypeError:
    # Fallback for infinite datasets
    num_images = getattr(testdataset, "cardinality", lambda: 1)().numpy()

avg_time = total_time / num_images if num_images > 0 else float('nan')

print(f"\n Total model inference time: {total_time:.3f} seconds")
print(f" Average time per image: {avg_time:.3f} seconds")

mean_loss = tf.reduce_mean(test_results[0])
print('The average loss is ', mean_loss)

mean_bpp = tf.reduce_mean(test_results[1])
print('The average bpp is ', mean_bpp)

mean_mse = tf.reduce_mean(test_results[2])
print('The average mse is ', mean_mse)

mean_psnr = tf.reduce_mean(test_results[3])
print('The average psnr is ', mean_psnr)

mean_ssim = tf.reduce_mean(test_results[4])
print('The average ssim is ', mean_ssim)

mean_msssim = tf.reduce_mean(test_results[5])
print('The average msssim is ', mean_msssim)

reconstructed_ds = test_results[6]

core_output = []

for i in range(len(reconstructed_ds)):
  core_out = reconstructed_ds[i]
  core = np.clip(core_out, a_min=0, a_max=255)
  core_output.append(core)

#from matplotlib.pyplot import imshow

#imshow(np.uint8(core_output[8]))

#imshow(np.uint8(original_ds[8]))

core_psnr = []
core_ssim = []
core_msssim = []

for i in range(len(original_ds)):
  c_psnr = tf.image.psnr(original_ds[i], np.float32(core_output[i]), max_val=255.)
  c_psnr = tf.cast(c_psnr, dtype=tf.float32)
  core_psnr.append(c_psnr)
  c_ssim = tf.image.ssim(original_ds[i], np.float32(core_output[i]), max_val=255.)
  c_ssim = tf.cast(c_ssim, dtype=tf.float32)
  core_ssim.append(c_ssim)
  c_msssim = tf.image.ssim_multiscale(original_ds[i], np.float32(core_output[i]), max_val=255.)
  c_msssim = tf.cast(c_msssim, dtype=tf.float32)
  core_msssim.append(c_msssim)

mean_psnr3 = tf.reduce_mean(core_psnr)
print('The average core PSNR is ', mean_psnr3)

mean_ssim3 = tf.reduce_mean(core_ssim)
print('The average core SSIM is ', mean_ssim3)

mean_msssim3 = tf.reduce_mean(core_msssim)
print('The average core MS-SSIM is ', mean_msssim3)

def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)

for i in range(len(core_output)):
  a = np.uint8(core_output[i])
  write_png("Give filepath", a)

for i in range(len(core_output)):
  a = np.uint8(original_ds[i])
  write_png("Give filepath", a)
