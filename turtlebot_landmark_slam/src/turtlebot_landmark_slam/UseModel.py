import torch
from PIL import Image
from torchvision import transforms
from network import ResNet18

CLASS_NAMES = {
    0: "Stop",
    1: "Turn right",
    2: "Turn left",
    3: "Ahead only",
    4: "Roundabout mandatory"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = ResNet18(num_classes=5).to(device)
checkpoint = torch.load("checkpoint/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])  # IMPORTANT
model.eval()

def classify_sign(crop):
    if not isinstance(crop, Image.Image):
        crop = Image.fromarray(crop)

    crop = crop.convert("RGB")
    x = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    return pred, CLASS_NAMES[pred]