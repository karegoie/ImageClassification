import torchvision.models as models
import torch.nn as nn

def inception(args):
        model = models.inception_v3(pretrained=True)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, args.classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,args.classes)

        return model