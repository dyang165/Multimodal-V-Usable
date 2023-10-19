import transformers
import sys
from datasets import *
from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.hooks import ModelWithHook

def collate_fn(batch, processor, num_classes, config):
    task = config["task"]
    drop_text, drop_images, drop_audio = None, None, None
    if 'drop_text' in config:
        drop_text = config["drop_text"]
    if 'drop_images' in config:
        drop_images = config["drop_images"]
    if 'drop_audio' in config:
        drop_audio = config["drop_audio"]    
 
    kwargs ={"padding":'max_length', "truncation": True, "return_tensors": 'pt'}
    if 'image' in batch[0] and not drop_images:
        images = [item['image'] for item in batch]
        kwargs['images'] = images
        print("drop_images")
    if 'text' in batch[0] and not drop_text:
        text = [item['text'] for item in batch]
        kwargs['text'] = text
    if 'audio' in batch[0] and not drop_audio:
        audio = [item['audio'] for item in batch]
        kwargs['audio'] = audio
    if 'max_length' in config:
        # No need to expand length for bert and vit
        kwargs['max_length'] = config['max_length']
    if 'truncation' in config:
        kwargs['truncation'] = config['truncation']
    inputs =  processor(**kwargs)
    #if drop_text:
    #    inputs = modality_dropout_text(inputs)
    #if drop_images:
    #    inputs = modality_dropout_image(inputs)

    # Process Labels
    if task == 'multilabel':
        labels = [torch.Tensor(item['labels']).long() for item in batch]
        labels = [torch.sum(F.one_hot(label, num_classes=num_classes), dim=0) for label in labels]
        labels = torch.stack(labels).float()
    elif task == 'multiclass':
        labels = torch.Tensor([item['labels'] for item in batch]).long()
    return inputs, labels

def modality_dropout_text(inputs):
    if 'text_inputs' in inputs:
        batch = inputs['text_inputs']
    else:
        batch = inputs
    batch['input_ids'] = torch.zeros_like(batch['input_ids'])
    batch['input_ids'][:,0] = torch.ones_like(batch['input_ids'])[:,0]*101
    batch['input_ids'][:,1] = torch.ones_like(batch['input_ids'])[:,1]*101
    batch['attention_mask'][:,2:] = torch.zeros_like(batch['attention_mask'][:,2:])
    batch['token_type_ids'] = torch.zeros_like(batch['token_type_ids'])
    if 'text_inputs' in inputs:
        inputs['text_inputs'] = batch
    else:
        inputs = batch
    return inputs

def modality_dropout_image(inputs):
    if 'image_inputs' in inputs:
        batch = inputs['image_inputs']
    else:
        batch = inputs
    batch['pixel_values'] = torch.ones_like(batch['pixel_values'])
    #batch['pixel_mask'] = torch.ones_like(batch['pixel_mask'])
    if 'image_inputs' in inputs:
        inputs['image_inputs'] = batch
    else:
        inputs = batch
    return inputs

def print_parameters(model):
    for name, param in model.named_parameters():
        prod = 1
        for i in param.shape:
            prod*=i
        print(name, prod)
 
def freeze_model(model, names, train_head = True):
    for name, param in model.named_parameters():
        if name in names:
            # VILT weights here were not loaded
            # We need to train them
            param.requires_grad = True
        elif 'head' in name:
            # Add option for freezing head for sanity
            param.requires_grad = train_head
        else:
            # These are vilt-loaded weights
            param.requires_grad = False       
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extend_pos_embeds(model, max_length):
    pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
    embedding_size = pretrained_pos_emb.weight.shape[1]
    extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
    model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(extended_pos_emb, freeze=False)
    model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
    model.max_position_embeddings = max_length

def get_dataset(dataset_name, split='train'):
    if split == 'train':
        if dataset_name == "mmimdb":
            return MMIMDB("train"), MMIMDB("dev")
        elif dataset_name == 'avmnist':
            return AVMNIST("train"), AVMNIST("dev")
        elif dataset_name == "crisis":
            return CRISIS("train"), CRISIS("dev") 
        elif dataset_name == "hateful":
            return HATEFUL("train"), HATEFUL("dev")
        elif dataset_name == 'n24':
            return N24News("train"), N24News("dev")
        elif dataset_name == 'n24h':
            return N24Headline("train"), N24Headline("dev") 
        elif dataset_name == 'food101':
            return Food101("train"), Food101("test")
    elif split == 'test':
        if dataset_name == "mmimdb":
            return MMIMDB("test")
        if dataset_name == 'crisis':
            return CRISIS("test") 
        if dataset_name == 'n24':
            return N24News("test")
        if dataset_name == 'n24h':
            return N24Headline("test")
        if dataset_name == 'food101':
            return Food101("test")

model_registry = {'vilt': 'dandelin/vilt-b32-mlm', 'bert': 'bert-base-uncased', 'vit':'google/vit-base-patch16-224'}

def get_model_and_processor(config):
    model_name = config["model"]
    pt_checkpoint = config["pt_checkpoint"] if 'pt_checkpoint' in config else None
    if pt_checkpoint in model_registry:
        pt_checkpoint = model_registry[pt_checkpoint]
    load_finetuned = "model_checkpoint" in config
    model = None
    processor = None
    if load_finetuned:
        print("Loading Trained Model")
        model = torch.load(config["model_checkpoint"], map_location = torch.device(0))
    if model_name == 'deit':
        if not load_finetuned:
            model = transformers.DeiTModel.from_pretrained(pt_checkpoint)
        processor = transformers.DeiTImageProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'vit': 
        if not load_finetuned:
            model = transformers.ViTModel.from_pretrained(pt_checkpoint)
        processor = transformers.ViTImageProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'bert': 
        if not load_finetuned: 
            model = transformers.BertModel.from_pretrained(pt_checkpoint)
        processor =  transformers.BertTokenizerFast.from_pretrained(pt_checkpoint)
    elif model_name == 'transformerapprox':
        text_pt_checkpoint = config["text_pt_checkpoint"]
        image_pt_checkpoint = config["image_pt_checkpoint"]
        if not load_finetuned:
            model = TransformerGaussianApprox(pt_checkpoint, config["num_classes"])
        processor = MergeProcessorVLLF(transformers.AutoProcessor.from_pretrained(text_pt_checkpoint),transformers.AutoProcessor.from_pretrained(image_pt_checkpoint))
    elif model_name == 'gaussianapprox':
        text_pt_checkpoint = config["text_pt_checkpoint"]
        image_pt_checkpoint = config["image_pt_checkpoint"]
        if not load_finetuned:
            model = GaussianApprox(pt_checkpoint, config["num_classes"])
        processor = MergeProcessorVLLF(transformers.AutoProcessor.from_pretrained(text_pt_checkpoint),transformers.AutoProcessor.from_pretrained(image_pt_checkpoint))


    elif model_name == 'vilt':
        # Can freeze here for linear probing
        if not load_finetuned:
            model = transformers.ViltModel.from_pretrained(pt_checkpoint)
            
            # Extend max length of text
            if config["max_length"] != None and config['max_length'] != 40:        
                max_length = config["max_length"]
                embedding_size = model.config.hidden_size
                pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
                extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
                model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(extended_pos_emb, freeze=False)
                model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
        processor = transformers.ViltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'viltnoimg':
        # Can freeze here for linear probing
        if not load_finetuned:
            model = transformers.ViltModelNoImg.from_pretrained(pt_checkpoint)
            
            # Extend max length of text
            if config["max_length"] != None:
                max_length = config["max_length"]
                embedding_size = model.config.hidden_size
                pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
                extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
                model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(extended_pos_emb, freeze=False)
                model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
        processor = transformers.ViltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'viltnotext':
        # Can freeze here for linear probing
        if not load_finetuned:
            model = transformers.ViltModelNoText.from_pretrained(pt_checkpoint)
            
            # Extend max length of text
            if config["max_length"] != None:
                max_length = config["max_length"]
                embedding_size = model.config.hidden_size
                pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
                extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
                model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(extended_pos_emb, freeze=False)
                model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
        processor = transformers.ViltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'viltnoinput':
        # Can freeze here for linear probing
        if not load_finetuned:
            model = transformers.ViltModelNoInput.from_pretrained(pt_checkpoint)
            
            # Extend max length of text
            if config["max_length"] != None:
                max_length = config["max_length"]
                embedding_size = model.config.hidden_size
                pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
                extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
                model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(extended_pos_emb, freeze=False)
                model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
        processor = transformers.ViltProcessor.from_pretrained(pt_checkpoint)

    elif model_name == 'roberta':
        if not load_finetuned:
            model = transformers.RobertaModel.from_pretrained(pt_checkpoint)
        processor = transformers.RobertaTokenizerFast.from_pretrained(pt_checkpoint)
    elif model_name == 'vitmae':
        if not load_finetuned:
            model = transformers.ViTMAEModel.from_pretrained(pt_checkpoint)
        processor =  transformers.ViTImageProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'tvlt':
        if not load_finetuned:
            model = transformers.TvltModel.from_pretrained(pt_checkpoint)
        processor = transformers.TvltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'tvltnoaudio':
        if not load_finetuned:
            model = transformers.TvltModelNoAudio.from_pretrained(pt_checkpoint)
        processor = transformers.TvltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'tvltnoimage':
        if not load_finetuned:
            model = transformers.TvltModelNoImage.from_pretrained(pt_checkpoint)
        processor = transformers.TvltProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'beit':
        if not load_finetuned:
            model = transformers.BeitForImageClassification.from_pretrained(pt_checkpoint).beit
        processor = transformers.BeitImageProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'beit_linprobe':
        if not load_finetuned:
            model = transformers.BeitForImageClassification.from_pretrained(pt_checkpoint).beit
            layer = model.encoder.layer[8]
            model = ModelWithHook(model, layer)
        processor =  transformers.BeitImageProcessor.from_pretrained(pt_checkpoint)
    elif model_name == 'vllf':
        text_pt_checkpoint = config['text_pt_checkpoint']
        image_pt_checkpoint = config['image_pt_checkpoint']
        if not load_finetuned:
            model = VLLF(text_pt_checkpoint, image_pt_checkpoint)
        processor = MergeProcessorVLLF(transformers.AutoProcessor.from_pretrained(text_pt_checkpoint),transformers.AutoProcessor.from_pretrained(image_pt_checkpoint))
    elif model_name == 'vllf-doublevilt':
        text_pt_checkpoint = config['text_pt_checkpoint']
        image_pt_checkpoint = config['image_pt_checkpoint']
        if not load_finetuned:
            model = VLLFDoubleVilt(config)
        processog = MergeProcessorDoubleVilt(transformers.AutoProcessor.from_pretrained(text_pt_checkpoint),transformers.AutoProcessor.from_pretrained(image_pt_checkpoint), config)
    
    elif model_name == 'resnet50':
        if not load_finetuned:
            model = ResNet50ModelWrapper("microsoft/resnet-50")
    if model == None and pt_checkpoint != None:
        # Default load
        model = transformers.AutoModel.from_pretrained(pt_checkpoint)
    if processor == None: 
        processor = transformers.AutoProcessor.from_pretrained(pt_checkpoint)
    if hasattr(model, 'module'):
        # Remove DataParallel Wrapper
        model = model.module
    if not load_finetuned:
        # Add Head
        if not 'head' in config:
            if 'hidden_size' in config:
                head = nn.Linear(config['hidden_size'], config["num_classes"])
            else:
                head = nn.Linear(model.config.hidden_size, config["num_classes"])
            dropout = config["dropout"] if "dropout" in config else 0.1
            apply_pooler = True
            average_pool = False
            if "apply_pooler" in config and config["apply_pooler"] == False:
                apply_pooler = False
            if "average_pool" in config and config["average_pool"] == True:
                average_pool = True    
            model = ModelWithHeadDropout(model, head, average_pool = average_pool, apply_pooling_layer=apply_pooler, dropout=dropout)
        else:
            if config['head'] == 'lf_mtl':
                model = LateFusionWithMultipleHeads(model, config["num_classes"]) 
            if config['head'] == 'lf_umt':
                model = LateFusionUMT(model, config['num_classes'], config['text_teacher_checkpoint'], config['image_teacher_checkpoint'])
            if config['head'] == 'ef_mtl':
                model = EarlyFusionWithMultipleHeads(model, config["num_classes"])
            if config['head'] == 'ef_mtl_av':
                model = EarlyFusionWithMultipleHeadsAV(model, config["num_classes"])
            if config['head'] == 'ef_mtl1':
                model = EarlyFusionWithMultipleHeads1(model, config["num_classes"])
            if config['head'] == 'ef_mtl2':
                model = EarlyFusionWithMultipleHeads2(model, config["num_classes"])
            if config['head'] == 'ef_mmt1':
                if config['push']:
                    text_teacher = config['text_teacher_checkpoint']
                    image_teacher = config['image_teacher_checkpoint']
                else:
                    text_teacher = config['teacher_checkpoint']
                    image_teacher = text_teacher
                model = EarlyFusionMMT1(model, config["num_classes"], text_teacher, image_teacher,  config['push']) 
            if config['head'] == 'ef_mmt2':
                if config['push']:
                    text_teacher = config['text_teacher_checkpoint']
                    image_teacher = config['image_teacher_checkpoint']
                else:
                    text_teacher = config['teacher_checkpoint']
                    image_teacher = text_teacher
                model = EarlyFusionMMT2(model, config["num_classes"], text_teacher, image_teacher,  config['push']) 
            if config['head'] == 'ef_push':
                model = EarlyFusionPush(model, config["num_classes"], push_modality = config["push_modality"])
            if config['head'] == 'ef_mask':
                model = EarlyFusionWithOneHead(model, config["num_classes"], modality = config["modality"]) 
            if config['head'] == 'resnet50':
                model = Resnet50Classifier(model, config['num_classes'])
            if config['head'] == 'bert':
                model = BertClassifier(model, config['num_classes'])
            if config['head'] == 'umt':
                model = LateFusionUMT(model, config["num_classes"])
    
    if load_finetuned:
        print("Loading Trained Model")
        state_dict = torch.load(config["model_checkpoint"], map_location = torch.device(0)).module.state_dict()
        pt_state_dict = model.state_dict()
        prefix = config["prefix"] if "prefix" in config else ""
        count = 0
        for pt_name in pt_state_dict:
            if prefix + pt_name in state_dict:
                count += 1
                pt_state_dict[pt_name] = state_dict[prefix + pt_name]
        print(f"Matched {count} / {len(pt_state_dict.keys())} params")
        model.load_state_dict(pt_state_dict)
                    
    return model, processor
