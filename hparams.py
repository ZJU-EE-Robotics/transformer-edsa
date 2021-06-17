################################
# Experiment Parameters        #
################################
epochs = 2000
iters_per_checkpoint = 1000
seed = 1234
cudnn_enabled = True
cudnn_benchmark = False
trans_type = "phn"

################################
# Data Parameters             #
################################
load_mel_from_disk = True
data_files = "filelists/data.csv"
training_files = "filelists/train_set.csv"
validation_files = "filelists/dev_set.csv"
test_files = "filelists/test_set.csv"
custom_files = "filelists/custom_set.csv"
dump = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/wavs"

################################
# Audio Parameters             #
################################

# if use tacotron 1's feature normalization
tacotron1_norm = False
preemphasis = 0.97
ref_level_db = 20.0
min_level_db = -100.0

sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
n_symbols = 35 if trans_type == "char" else 78

# body
d_embed = 512
d_mel = 80
d_model = 512
d_inner = 2048
n_head = 8
n_layers = 6
n_position = 1500
n_frames_per_step = 1   # TODO: currently, only 1 is supported
max_decoder_steps = 1500
stop_threshold = 0.5
infer_trim = 2

# Encoder prenet parameters
eprenet_chans = 512
eprenet_kernel_size = 5
eprenet_n_convolutions = 3

# Decoder prenet parameters
dprenet_size = 256

# Mel-post processing network parameters
dpostnet_chans = 512
dpostnet_kernel_size = 5
dpostnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 0.02   # 0.0442 is 1 / sqrt(d_model)
adam_beta1 = 0.9
adam_beta2 = 0.98
adam_eps = 1e-06       # 1e-06 for amp and 1e-09 for regular
weight_decay = 0.0
warmup_step = 4000
grad_clip_thresh = 2.0
batch_size = 8
accum_size = 8
