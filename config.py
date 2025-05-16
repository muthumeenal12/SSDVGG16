import torch

BATCH_SIZE = 16  # Increase / decrease according to GPU memory.
RESIZE_TO = 640  # Resize the image for training and transforms.
NUM_EPOCHS = 75  # Number of epochs to train for.
NUM_WORKERS = 2  # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = '/content/person_dataset/Train/Train/JPEGImages'
# Validation images and XML files directory.
VALID_DIR = '/content/person_dataset/Val/Val/JPEGImages'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'person'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'

# Optimizer and Scheduler parameters
LEARNING_RATE = 0.0005  # Base learning rate
MOMENTUM = 0.9
NESTEROV = True

SCHEDULER_MILESTONES = [45, 60]  # Epochs where LR will decay
SCHEDULER_GAMMA = 0.1  # LR decay factor
