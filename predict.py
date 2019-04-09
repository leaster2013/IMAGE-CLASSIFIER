def main():
    input_arguments = input_argparse()
    im = Image.open(input_arguments.input_image_path)
    device = train.device_in_use(gpu_ind=input_arguments.gpu)
    label_to_name_json = cat_to_name_conv()
    model = train.load_checkpoint(checkpoint_loc = input_arguments.checkpoint_name+'.pth')
    probability, prediction = predict(image_path = im, model = model,topk=input_arguments.top_k , device = device)
    probability = probability.to('cpu')
    prediction = prediction.to('cpu')
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    prediction.numpy()[0] = [idx_to_class[x] for x in prediction.numpy()[0]]
    top_classes = [label_to_name_json[str(x)] for x in  prediction.numpy()[0]]
    top_probabilities = probability.numpy()[0]
    print('predicted flower name :'+str(top_classes[0]))
    print('PROBABILITY'+' '+'PREDICTION')
    for probability, prediction in zip(top_probabilities ,top_classes):
        print(str(probability)+' : '+str(prediction))
    os.environ['QT_QPA_PLATFORM']='offscreen'
    check()
def check():
    c = 1
    path = 'flowers/test/1/image_06743.jpg'
    probabilities = predict(path, model)
    image = process_image(path)

    i = imshow(image, ax = plt)
    i.axis('off')
    i.title(cat_to_name[str(c)])
    i.show()
    
    a = np.array(probabilities[0][0])
    index = [cat_to_name[str(c+ 1)] for c in np.array(probabilities[1][0])]
    fig,ax = plt.subplots(figsize=(8,3))
    ax.bar(np.arange(len(index)), a)
    ax.set_xticks(ticks = np.arange(len(index)))
    ax.set_xticklabels(index)
    ax.set_yticks([0.2,0.4,0.6,0.8,1])
    plt.show()
def cat_to_name_conv():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
def process_image(image):
    pil_img = Image.open(image)
   
    adj = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adj(pil_img)
    
    return img_tensor
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
def predict(image_path, model,device = 'gpu',topk=5):
    model.to(device)
    i = process_image(image_path)
    i = i.unsqueeze_(0)
    i = i.float()
    
    with torch.no_grad():
        i = model.forward(i.cuda())
        
    probability = F.softmax(i.data,dim=1)
    
    return probability.topk(topk)
    return probability,prediction
def input_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path', type=str, default= 'flowers/test/10/image_07090.jpg',
                        help = 'set the directory where the image is present')
    parser.add_argument('checkpoint_name', type=str, default='checkpoint',
                        help='checkpoint name')
    parser.add_argument('--top_k', type = int, default=5,
                        help = 'set the number of top matching results')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help= 'label to name json filename')
    parser.add_argument('--gpu', action = 'store_true',
                        help='Enable cuda')
    return parser.parse_args()
if __name__ == '__main__':
    main()