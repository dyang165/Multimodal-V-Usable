import torch.nn as nn
import torch
import copy

class Pooler(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, tensor):
        pooled_output = self.dense(tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  

class ModelWithHead(nn.Module):
    def __init__(self, model, head, add_pooling_layer=True):
        super().__init__()
        self.model = model
        self.head = head
        self.apply_pooling_layer = apply_pooling_layer
        self.pooler = Pooler()
   
    def forward(self, **x):
        x = self.model(**x)
        if self.apply_pooling_layer:
            # Applies additional dense layer + tanh
            if hasattr(x, 'pooler_output'):
                x = self.dropout(x.pooler_output)
                x = self.head(x)
            else:
                x = self.pooler(x.last_hidden_state)
                x = self.dropout(x)
                x = self.head(x)
        else:
            x = self.dropout(x.last_hidden_state[:,0,:])    
            x = self.head(x)
        return x
    
    
class ModelWithHeadDropout(nn.Module):
    def __init__(self, model, head, apply_pooling_layer=True, average_pool = False, dropout=0.1):
        super().__init__()
        assert((apply_pooling_layer and average_pool) != True) 
        self.model = model
        self.head = head
        self.apply_pooling_layer = apply_pooling_layer
        self.average_pool = average_pool
        self.pooler = Pooler()
        self.dropout = nn.Dropout(dropout)    
    
    def legacy_forward(self, x):
        if hasattr(self, 'add_pooling_layer'):
            if self.add_pooling_layer:
                x = self.head(x.pooler_output)
            else:
                x = self.head(x.last_hidden_state[:,0,:])
        else:
            x = self.head(x.pooler_output)
        return x
  
    def forward(self, **x):
        x = self.model(**x)
        if not hasattr(self,'apply_pooling_layer'):
            return self.legacy_forward(x)
        if self.average_pool: 
            x = torch.mean(x.last_hidden_state, dim=1)
            #x = self.pooler(x)
            x = self.dropout(x)
            x = self.head(x) 
        else:
            if self.apply_pooling_layer:
                # Applies additional dense layer + tanh
                if hasattr(x, 'pooler_output'):
                    x = self.dropout(x.pooler_output)
                    x = self.head(x)
                else:
                    x = self.pooler(x.last_hidden_state[:,0,:])
                    x = self.dropout(x)
                    x = self.head(x)
            else:
                x = self.dropout(x.last_hidden_state[:,0,:])    
                x = self.head(x)
        return x


class LateFusionWithMultipleHeads(nn.Module):
    def __init__(self, model, num_classes, text_dim = 768, image_dim = 768,  dropout=0.1):
        super().__init__()
        self.model = model
        self.head_text = nn.Linear(text_dim, num_classes)
        self.head_image = nn.Linear(image_dim, num_classes) 
        self.head_mm = nn.Linear(text_dim + image_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_dim = text_dim
        self.image_dim = image_dim
 
    def forward(self, **x):
        x = self.model(**x)
        x = self.dropout(x.pooler_output)
        
        x_text = x[:,:self.text_dim]
        x_text = self.head_text(x_text)
        
        x_image = x[:,self.text_dim:]
        x_image = self.head_image(x_image)

        x = self.head_mm(x)
        return x_text, x_image, x

class LateFusionUMT(nn.Module):
    def __init__(self, model, num_classes, text_pt, image_pt, text_dim = 768, image_dim = 768,  dropout=0.1):
        super().__init__()
        self.model = model
        self.head_mm = nn.Linear(text_dim + image_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.text_teacher = torch.load(text_pt).module.model
        self.image_teacher = torch.load(image_pt).module.model
        self.freeze(self.text_teacher)
        self.freeze(self.image_teacher)
         
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def forward(self, **inputs):
        x = self.model(**inputs)
        x = self.dropout(x.pooler_output)
        
        x_text = x[:,:self.text_dim]
        text_teacher_out = self.text_teacher(**inputs['text_inputs']).pooler_output 
        x_text = torch.sum(torch.norm(text_teacher_out-x_text, dim=1)**2)
         
        x_image = x[:, self.text_dim:]
        image_teacher_out = self.image_teacher(**inputs['image_inputs']).pooler_output.squeeze()
        x_image = torch.sum(torch.norm(image_teacher_out- x_image, dim=1)**2)

        x = self.head_mm(x)
        return x_text, x_image, x


class EarlyFusionMMT(nn.Module):
    def __init__(self, model, num_classes, text_pt, image_pt, push, hidden_dim = 768, text_len = 520,  dropout=0.1):
        super().__init__()
        self.model = model
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.hidden_dim = hidden_dim
        self.text_teacher = torch.load(text_pt).module.model
        self.image_teacher = torch.load(image_pt).module.model
        self.freeze(self.text_teacher)
        self.freeze(self.image_teacher)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.push = push
        self.mse = nn.MSELoss()
        self.text_len = text_len
        
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def average_pool(self, x):
        return torch.mean(x, dim=1)    
    
    def forward(self, **inputs):
        # Get model outputs
        x = self.model(mask_modality_interaction=True, **inputs).last_hidden_state
        x = self.dropout(x)
 
        x_text = x[:,:self.text_len,:]
        text_teacher_out = self.text_teacher(**inputs).last_hidden_state
        if self.push:
            x_text = self.mse(text_teacher_out[:,:self.text_len,:],self.projector_text(x_text))
        else:
            x_text = self.mse(self.average_pool(text_teacher_out),self.projector_text(self.average_pool(x_text)))
         
        x_image = x[:, self.text_len:, :]
        if self.push:
            image_teacher_out = self.image_teacher(**inputs).last_hidden_state
            x_image = self.mse(image_teacher_out[:,self.text_len:,:],self.projector_image(x_image))
        else:
            image_teacher_out = text_teacher_out.clone()
            x_image = self.mse(self.average_pool(image_teacher_out),self.projector_image(self.average_pool(x_image)))

        x_mm = self.model(mask_modality_interaction=False, **inputs).last_hidden_state 
        x_mm = self.head_mm(self.average_pool(self.dropout(x_mm)))
        return x_text, x_image, x_mm


class EarlyFusionMMT1(nn.Module):
    def __init__(self, model, num_classes, text_pt, image_pt, push, hidden_dim = 768, text_len = 520,  dropout=0.1):
        super().__init__()
        self.model = model
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.hidden_dim = hidden_dim
        #self.text_teacher = torch.load(text_pt).module.model
        #elf.image_teacher = torch.load(image_pt).module.model
        #self.freeze(self.text_teacher)
        #self.freeze(self.image_teacher)
        #self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.push = push
        self.mse = nn.MSELoss()
        self.text_len = text_len
        
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def average_pool(self, x):
        return torch.mean(x, dim=1)    
    
    def forward(self, **inputs):
        # Get model outputs
        unimodal_outputs = self.model(mask_modality_interaction=True, **inputs).last_hidden_state
        multimodal_outputs = self.model(**inputs).last_hidden_state
        unimodal_outputs = self.dropout(unimodal_outputs)
        multimodal_outputs = self.dropout(multimodal_outputs)
        multimodal_target = self.average_pool(multimodal_outputs).clone().detach()
 

        text_output = self.average_pool(unimodal_outputs[:,:self.text_len,:])
        image_output = self.average_pool(unimodal_outputs[:,self.text_len:,:])
        
        text_reg = self.mse(text_output, multimodal_target)
        image_reg = self.mse(image_output, multimodal_target)

        x_mm = self.head_mm(self.average_pool(multimodal_outputs)) 

        return text_reg, image_reg, x_mm 

class EarlyFusionMMT2(nn.Module):
    def __init__(self, model, num_classes, text_pt, image_pt, push, hidden_dim = 768, text_len = 520,  dropout=0.1):
        super().__init__()
        self.model = model
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.hidden_dim = hidden_dim
        #self.text_teacher = torch.load(text_pt).module.model
        #elf.image_teacher = torch.load(image_pt).module.model
        #self.freeze(self.text_teacher)
        #self.freeze(self.image_teacher)
        #self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.push = push
        self.mse = nn.MSELoss()
        self.text_len = text_len
        
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def average_pool(self, x):
        return torch.mean(x, dim=1)    
    
    def forward(self, **inputs):
        # Get model outputs
        unimodal_outputs = self.model(mask_modality_interaction=True, **inputs).last_hidden_state
        unimodal_outputs = self.dropout(unimodal_outputs)
        multimodal_target = self.average_pool(unimodal_outputs).clone().detach()
 

        text_output = self.average_pool(unimodal_outputs[:,:self.text_len,:])
        image_output = self.average_pool(unimodal_outputs[:,self.text_len:,:])
        
        text_reg = self.mse(text_output, multimodal_target)
        image_reg = self.mse(image_output, multimodal_target)

        x_mm = self.head_mm(self.average_pool(unimodal_outputs)) 

        return text_reg, image_reg, x_mm 



class EarlyFusionPush(nn.Module):
    def __init__(self,model, num_classes, push_modality, text_len = 520, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.model = model
        self.head = nn.Linear(hidden_dim, num_classes)
        self.push_modality = push_modality
        self.text_len = text_len
        self.dropout = nn.Dropout(dropout)
    
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        x = self.model(**batch).last_hidden_state
        x = self.dropout(x)
        if self.push_modality == 'text':
            x = self.average_pool(x[:,:self.text_len,:])
        elif self.push_modality == 'image':
            x = self.average_pool(x[:,self.text_len:,:])
        else:
            print("Error push modality is wrong")
        x = self.head(x)
        return x

class EarlyFusionWithMultipleHeadsAV(nn.Module):
    def __init__(self, model, num_classes, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.model = model
        self.head_audio = nn.Linear(hidden_dim, num_classes)
        self.head_image = nn.Linear(hidden_dim, num_classes) 
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
 
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        x = self.model(mask_modality_interaction=True,**batch)
        x_image = x.last_pixel_hidden_state
        x_image = self.dropout(x_image)
        x_image = self.average_pool(x_image)
        x_image = self.head_image(x_image)
        
        x_audio = x.last_audio_hidden_state
        x_audio = self.average_pool(x_audio)
        x_audio = self.head_audio(x_audio)

        x = self.model(mask_modality_interaction=False, **batch).last_hidden_state
        x = self.average_pool(x)
        x = self.head_mm(x)
        return x_image, x_audio, x

class EarlyFusionWithMultipleHeads(nn.Module):
    def __init__(self, model, num_classes, text_len = 520, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.model = model
        self.head_text = nn.Linear(hidden_dim, num_classes)
        self.head_image = nn.Linear(hidden_dim, num_classes) 
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_len = text_len
 
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        x = self.model(mask_modality_interaction=True,**batch).last_hidden_state 
        x = self.dropout(x)
        x_text = x[:,:self.text_len,:]
        x_text = self.average_pool(x_text)
        x_text = self.head_text(x_text)
        
        x_image = x[:,self.text_len:,:]
        x_image = self.average_pool(x_image)
        x_image = self.head_image(x_image)

        x = self.model(mask_modality_interaction=False, **batch).last_hidden_state
        x = self.average_pool(x)
        x = self.head_mm(x)
        return x_text, x_image, x

class EarlyFusionWithMultipleHeads2(nn.Module):
    def __init__(self, model, num_classes, text_len = 520, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.model = model
        self.head_text = nn.Linear(hidden_dim, num_classes)
        self.head_image = nn.Linear(hidden_dim, num_classes) 
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_len = text_len
 
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        x = self.model(mask_modality_interaction=True, **batch).last_hidden_state
        x = self.dropout(x)
        x = self.average_pool(x)
        x = self.head_mm(x)
        return x

class EarlyFusionWithMultipleHeads1(nn.Module):
    def __init__(self, model, num_classes, text_len = 520, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.model = model
        self.head_text = nn.Linear(hidden_dim, num_classes)
        self.head_image = nn.Linear(hidden_dim, num_classes) 
        self.head_mm = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_len = text_len
 
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        x = self.model(mask_self_interaction=True, **batch).last_hidden_state
        x = self.dropout(x)
        x = self.average_pool(x)
        x = self.head_mm(x)
        return x
        #return 0, 0, x

class EarlyFusionWithOneHead(nn.Module):
    def __init__(self, model, num_classes, modality = None, text_len = 520, hidden_dim=768, dropout=0.1):
        super().__init__()
        assert(modality in ["text","image", None])
        self.model = model
        self.modality = modality
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)    
        self.text_len = text_len
 
    def average_pool(self, x):
        return torch.mean(x, dim=1)

    def forward(self, **batch):
        if self.modality == 'text': 
            x = self.model(mask_modality_interaction=True,**batch).last_hidden_state 
            x_text = x[:,:self.text_len,:]
            x = self.average_pool(x_text)
        elif self.modality == 'image':
            x = self.model(mask_modality_interaction=True,**batch).last_hidden_state 
            x_image = x[:,self.text_len:,:]
            x = self.average_pool(x_image)
        else:
            x = self.model(**batch).last_hidden_state 
            x = self.average_pool(x)
        x = self.head(x)
        return x


class Resnet50Classifier(nn.Module):
    def __init__(self, model, num_classes, dim = 2048):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(dim, num_classes)
         
    def forward(self, **x):
        x = self.model(**x).pooler_output.squeeze()
        output = self.classifier(x)
        return output 

class BertClassifier(nn.Module):
    def __init__(self, model, num_classes, has_cls = False, hidden_dim = 512):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(hidden_dim*2, num_classes)
         
    def forward(self, **x):
        x = self.model(**x).last_hidden_state
        cls_rep = x[:,0]
        mean_rep = torch.mean(x[:, 1:, :], dim=1)
        combined  = torch.cat((cls_rep, mean_rep), dim=-1).float()
        output = self.classifier(combined)
        return output 
