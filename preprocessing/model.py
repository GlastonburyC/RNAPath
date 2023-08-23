import torch
import torchvision.models as torchvision_models
import os



class FeaturesExtraction_IMAGENET:

    def __init__(self):
        self.model = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.DEFAULT)
        self.transform = torchvision_models.ResNet50_Weights.DEFAULT.transforms()

    def extractFeatures(self, img, device):        
        with torch.no_grad():
            feats = self.model(img.to(device)).clone()
            feats = feats.cpu().detach().numpy()
        return feats


class FeaturesExtraction_Vit:

    def __init__(self, pretrained_weights=None, checkpoint_key='teacher'):
        

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # build model
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=True, img_size=[224])
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(self.device)
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            print('No pretrained weights found. Using default weights.')


    def extractFeatures(self, img, device):

        with torch.no_grad():
            feats = self.model(img.to(device)).clone()
            feats = feats.cpu().detach().numpy()
        return feats

