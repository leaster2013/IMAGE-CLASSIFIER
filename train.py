import numpy as np
from torchvision import datasets,transforms, models
import torch, os, json, argparse
from torch import nn,optim
from collections import OrderedDict
def main():
    input_arguments = input_argparse()
    device = device_in_use(gpu_ind = input_arguments.gpu)
    model_name = input_arguments.arch
    model = model(model_name = model_name, hidden_units=input_arguments.hidden_units)
    train_model(data_dir=input_arguments.data_dir, model = model, device= device, model_mode = model_mode,
                learning_rate = input_arguments.learning_rate,model_name= model_name, hidden_units = input_arguments.hidden_units)
def model(model_name = 'vgg16', hidden_units = 512):
    model = getattr(models, model_name)(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU(inplace=True)),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model
def save_checkpoint():
    model.class_to_idx = image_datasets['test'].class_to_idx
    
    cp = {'state_dict' : model.state_dict(),
                  'optimizer_state' : optimizer.state_dict(),
                  'class_to_idx': image_datasets['test'].class_to_idx,
                  'model_name': 'vgg16',
                  'classifier':model.classifier
                 }
    torch.save(cp, 'cp.pth')

def load_checkpoint():
    cp = torch.load('cp.pth')
    model = getattr(models, cp['model_name'])(pretrained=True)
    model.classifier = cp['classifier']
    model.load_state_dict(cp['state_dict'])
    model.class_to_idx = cp['class_to_idx']
    return model
def run_accuracy_check(device,model,inputs):
    c = 0
    t = 0
    model.to(device)
    with torch.no_grad():
    for x in dataloaders['test']:
        i, j = x
        i, j= i.to('cuda'), j.to('cuda')
        i = model.forward(i)
        _, predicted = torch.max(i.data, 1)
        t = t+ j.size(0)
        c = c+ (predicted == j).sum().item()

    print("The network's accuracy on the %d test images: %d %%" % (t, (100 * correct / total)))
def validation(model, inputs, criterion, device):
    tloss = 0 
    acc = 0
    for i, j in inputs:
        i, j = i.to(device), j.to(device)
        i = model.forward(i)
        tloss = tloss+criterion(i,j).item()
        
        ps = torch.exp(i)
        e = (j.data == ps.max(dim=1)[1])
        
        acc = acc+ e.type(torch.FloatTensor).mean()
        
    return tloss,acc
def train_model(device, model, model_mode, model_name, hidden_units ,data_dir='flowers/', step = 0 ,epochs = 3, print_every = 40, learning_rate = 0.001):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(40),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    image_datasets = {x : datasets.ImageFolder(root= 'flowers'+ '/' + x,transform=data_transforms[x])
                      for x in list(data_transforms.keys())                  
                     }
    dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x],shuffle=True, batch_size=32) 
                   for x in list(data_transforms.keys())
                  }
    count = 0
    epochs = 4
    print_every = 30
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    for e in range(epochs):
        model.train()
        rloss = 0
        rcorrects = 0
        for i, j in dataloaders['train']:
            i, j = i.to(device), j.to(device)
            count=count + 1
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                i = model.forward(i)
                _, predicted = torch.max(i.data, 1)
                loss = criterion(i,j)
                loss.backward()
                optimizer.step()
                rloss = rloss+ loss.item()
                rcorrects = rcorrects + torch.sum(predicted == j.data)
            if count % print_every == 0:
                model.eval()
                with torch.no_grad():
                    tloss,acc = validation(model=model,inputs=dataloaders['valid'],criterion=criterion)
                    print("Epoch : {}/{}".format(e+1, epochs),
                      "Training loss: {:.3f}".format(rloss/print_every),
                      "Training Accuracy: {:.3f}".format(rcorrects.double()/len(dataloaders['train'])),
                      "Test loss : {:.3f}".format(tloss/len(dataloaders['valid'])),
                      "Test Accuracy : {:.3f}".format(acc/len(dataloaders['valid']))
                          )
                rloss = 0
                rcorrects = 0
                model.train()
    run_accuracy_check(device= device, model=model , inputs = dataloaders[model_mode[1]])
    print("Saving the model...")
    save_checkpoint(model = model,optimizer= optimizer,epochs = epochs,hidden_units =hidden_units,
                    image_input = image_datasets[model_mode[2]], learning_rate = learning_rate, model_name = model_name)
def device_in_use(gpu_ind= True):
    if gpu_ind and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
def input_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default= 'flowers/',
                        help = 'set the directory where the data is present')
    parser.add_argument('--save_dir',type=str, default='checkpoints/',
                        help = 'directory where the checkpoint will be saved')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='select the pretrained model')
    parser.add_argument('--learning_rate', type=float , default=0.001,
                        help = 'set the training model learning rate')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='set the training model\'s hidden units')
    parser.add_argument('--epochs', type=int, default=3,
                        help='set the training model epoch')
    parser.add_argument('--gpu', action = 'store_true',
                        help='Enable cuda')
    return parser.parse_args()
if __name__ == '__main__':
    main()