import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F


# Transformation just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Save model
def save_model(model, path):
    if not os.path.exists(path):
        dir, file = os.path.split(path)
        os.mkdir(dir) 
    torch.save(model.state_dict(), path)
    print(f"Model save to {path}")
    
# Remove the results with low confident rate 
def remove_under_confident(prediction: dict, score: float):
    """prediction is a dictionary with keys:
    boxes: tensor(n, 4)
    labels: tensor(n, 1)
    scores: tensor(n, 1)
    we need to remove all the results with scores under confident score

    Args:
        predictions (dict): _description_
    """
    results = prediction["scores"]
    criteria_meet = [results > score]
    for key, item in prediction.items():
        prediction[key] = item[criteria_meet]
    return prediction

# Draw bboxes and save to a directory
def test_visualization(image, pred_boxes, pred_labels, img_path, colors="blue"):
    """Visualize result of the model after the image pass through model

    Args:
        image (tensor): image converted to tensor type
        pred_boxes (list): list of predict box from predict_image
        pred_labels (list): list of string labels draw near box
        img_path (string): where to save the result image
        colors (str, optional): _description_. Defaults to "blue".
    """
    pred_boxes = torch.tensor(pred_boxes)
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors=colors)
    dir, file = os.path.split(img_path)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except:
            pass
    # Save the output image
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(img_path)
    print(f"Results save completed at {img_path}.")
    
def pick_image_example(dir):
    """Pick random image from a directory

    Args:
        dir (string): the directory want to pick

    Returns:
        PIL.Image, tensor_image, image_name: an PIL Image object
    """
    
    for root, dirs, files in os.walk(dir):
        while True:
            img_file = random.choice(files)
            img_path = os.path.join(dir, img_file)
            try:
                image = Image.open(img_path)
                break
            except:
                continue
        break
    image = F.pil_to_tensor(image)                  # convert to tensor shape (3,640,640)
    image_tensor = F.convert_image_dtype(image)     # covert to type for the model
    
    return image, image_tensor, img_file

