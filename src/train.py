from util import load_data
from models import ResNetCallig, VGG16, Model

if __name__ == '__main__':
    datasets = load_data()
    vgg_models = [Model(VGG16, data) for data in datasets]
    resnet_models = [Model(ResNetCallig, data) for data in datasets]
    
    for model in resnet_models:
        model.train()
        model.eval_model()
        model.save_model()
        