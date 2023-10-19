import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal
import copy

#class PosteriorGaussian(nn.Module):
#    def __init__(self, mtl_checkpoint, num_classes):
#        self.mean = nn.Parameter(torch.zeros(768), requires_grad=True)
#        self.cov = nn.Parameter(torch.eye(768), requires_grad=True)
#        self.mvn = []
#        for i in range(num_classes):
#            self.mvn.append(MultivariateNormal(loc=mean.view(1,2),
#                         scale_tril=cov.view(-1, 2, 2)))
#        self.multimodal_mtl_model = torch.load(checkpoint)
    
#    def forward(self,**inputs): 
#        bs, num_samples, hidden_size = x.shape
#        unimodal_logits, unimodal_logits = 
#        multimodal_rep =  
 
class GaussianApprox(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super().__init__()
        self.MTL_net = torch.load(checkpoint, map_location = torch.device(0))
        self.head_image = self.MTL_net.module.head_image
        self.head_mm = self.MTL_net.module.head_mm
        self.MTL_net = self.MTL_net.module.model
        self.device = list(self.MTL_net.parameters())[0].get_device()
        #self.approx_net = copy.deepcopy(self.MTL_net.module.model.img_encoder)
        
        self.num_classes = num_classes

            
        self.shared_layer = nn.Sequential(
            nn.Linear(769,769),
            nn.GELU(),
            nn.Dropout(),
        )

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(769, 769),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(769, 768),
        )
        
        self.cov_diag_layer = nn.Sequential(
            nn.Linear(769, 769),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(769, 768),
            nn.Softplus(),  # enforces positivity
        )
 
        self.freeze(self.MTL_net)
        self.freeze(self.head_mm)
        self.freeze(self.head_image)
        self.softmax = nn.Softmax(dim=1)
        self.mse = torch.nn.MSELoss()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad=False

    def normalize(self, x):
        pass

    def forward(self, **mtl_inputs):
        image_inputs = mtl_inputs["image_inputs"]
        vllf  = self.MTL_net(**mtl_inputs).pooler_output
        text_inputs, image_inputs = vllf[:,:768], vllf[:,768:]
        multimodal_probs = self.softmax(self.head_mm(vllf)) #p(y|x_1,x_2)
        unimodal_probs = self.softmax(self.head_image(image_inputs)) #p(y|x_1)
        ratio = torch.divide(multimodal_probs,unimodal_probs) #p(y|x_1,x_2)/p(y|x_1)
        bs,n = ratio.shape
        preds = []
        for i in range(self.num_classes):
            inputs = torch.cat([torch.ones(bs).to(self.device).view(bs,1)*i, image_inputs], dim=1) # cat([y,x1])
            inputs = self.shared_layer(inputs).view(bs, 769)
            mean_preds = self.mean_layer(inputs).view(bs,768)
            #cov_factor_preds = self.cov_factor_layer(inputs).view(bs,768,1)
            cov_diag_preds = torch.diag_embed(self.cov_diag_layer(inputs).view(bs,768))
            normal_dist = MultivariateNormal(mean_preds, cov_diag_preds)
            #normal_dist = LowRankMultivariateNormal(mean_preds, cov_factor_preds, cov_diag_preds)
            log_prob = normal_dist.log_prob(text_inputs).view(bs,1)  #log(p(x2|x_1,y))
            preds.append(torch.exp(log_prob))
        preds = torch.cat(preds, dim=1) # bs x num_classes
        marginal = torch.sum(log_prob + torch.log(unimodal_probs), dim=1) # log p(x_2|x_1) = log (p(x_2|x_1,y) dot p(y|x1))
        preds = preds - marginal.view(bs,1) # log(preds/marginal) with broadcasting
        return self.mse(torch.log(ratio), preds), preds

class TransformerApprox(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super().__init__()
        self.MTL_net = torch.load(checkpoint, map_location = torch.device(0))
        self.approx_net = copy.deepcopy(self.MTL_net.module.model.img_encoder)
        self.regression_head = nn.Linear(768, num_classes)
        self.freeze(self.MTL_net)
        self.softmax = nn.Softmax(dim=1)
        self.mse = torch.nn.MSELoss()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad=False

    def forward(self, **mtl_inputs):
        image_inputs = mtl_inputs["image_inputs"]
        _, unimodal_logits, multimodal_logits  = self.MTL_net(**mtl_inputs) 
        ratio = torch.divide(self.softmax(multimodal_logits),self.softmax(unimodal_logits))
        ratio = torch.log(ratio)
        
        approx_out = self.approx_net(**image_inputs).pooler_output
        pred = self.regression_head(approx_out)
        return self.mse(ratio, pred), pred
       


class TransformerGaussianApprox(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super().__init__()
        self.MTL_net = torch.load(checkpoint, map_location = torch.device(0)).module.model
        self.model = copy.deepcopy(self.MTL_net.img_encoder)
        self.mean_head = nn.Linear(768, 768)
        self.cov_head = nn.Sequential(nn.Linear(768,768), nn.Softplus())
        self.freeze(self.MTL_net)
        self.softmax = nn.Softmax(dim=1)
        self.mse = torch.nn.MSELoss()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad=False

    def forward(self, **mtl_inputs):
        image_inputs = mtl_inputs["image_inputs"]
        vllf  = self.MTL_net(**mtl_inputs).pooler_output 
        text_feats, image_feats = vllf[:,:768],vllf[:,768:]
        
        out = self.model(**image_inputs).pooler_output

        mean = self.mean_head(out)
        cov = self.cov_head(out)
        bs, hidden_dim = mean.shape
        normal = MultivariateNormal(mean, torch.diag_embed(cov))
        log_prob = normal.log_prob(text_feats)
        
        return -log_prob.mean(), None

