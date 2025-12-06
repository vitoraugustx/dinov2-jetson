import torch
from PIL import Image
from torchvision import transforms

from transformers import Dinov2Model
# Load a pre-trained DINOv2 model
dinov2_backbone = Dinov2Model.from_pretrained("facebook/dinov2-base-patch16-224")

# Example input image
image_path = "./images/0000000193.png"
image = Image.open(image_path).convert("RGB")

# Preprocessing (adjust as per your model's requirements)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

# Initialize segmentation head (adjust in_channels and num_classes)
segmentation_head = SimpleSegmentationHead(in_channels=768, num_classes=21) # Example values

# Perform inference
with torch.no_grad():
    features = dinov2_backbone(input_tensor).last_hidden_state
    # Reshape features to a spatial representation suitable for the head
    # This step depends on how your DINOv2 features are structured (e.g., tokens)
    # and what kind of segmentation head you are using.
    # For simple heads, you might need to reshape tokens to a grid.
    # Example: features = features[:, 1:].reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    
    segmentation_logits = segmentation_head(features)
    predicted_mask = torch.argmax(segmentation_logits, dim=1).squeeze(0) # Get class predictions