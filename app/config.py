# path to your own data and coco file
TRAIN_DATA_DIR = "data/train"
TRAIN_COCO = "data/train/_annotations.coco.json"
EVAL_DATA_DIR = "data/valid"
EVAL_COCO = "data/valid/_annotations.coco.json"

# path to save model, image
MODEL_PATH = "models/version-1.pth"
TEST_IMG_DIR = "result-images"
TEST_DATA_DIR = "data/test"

# Batch size
BATCH_SIZE = 4

# Params for dataloader
SHUFFLE_DL = True
NUM_WORKERS_DL = 4

# Params for training
NUM_CLASSES = 2     # Two classes; Only target class or background
NUM_EPOCHS = 1

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005

# Params for evaluation
CONFIDENT_SCORE = 0.8
