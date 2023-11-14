from util import load_data
from models import ResNetCallig, VGG16, Model
from torchvision import transforms

some_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Random rotation within ±10 degrees
    transforms.RandomHorizontalFlip(),      # Random horizontal flip
    transforms.RandomCrop(64, padding=4),   # Random crop with padding of 4 pixels
    transforms.ToTensor(),                  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))   # Normalize the pixel values
    # add gaussian, noise from util
])

if __name__ == '__main__':
    datasets = load_data(transform = some_transforms)
    vgg_models = [Model(VGG16, data) for data in datasets]
    resnet_models = [Model(ResNetCallig, data) for data in datasets]
    
    for model in resnet_models:
        model.train()
        model.eval_model()
        model.save_model()