from data.wiki import WikiArt
from stylenet import StyleNet
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import presets_orig
##conv part fine tuning
##parameters
num_style=20
crop_size=224
learning_rate=0.001
num_epochs=100
num_batch=64
wiki_csv="./data/wiki.csv"
img_dir="/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/"

##model
netname='vgg16' #original conv - [4096, 10000]
mlp=[[4096,1024],[1024,512],[512,20]]
dropout=[None,None,None]
activations=['relu','relu','relu']


##device 
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")



##loss softmax

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y,_) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y,_) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            current = (batch + 1) * len(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f"{current:>5d}/{size:>5d}")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    #build model
    model=StyleNet(name=netname,mlp=mlp,dropout=dropout,activations=activations).to(device)
    model.load_state_dict(torch.load("./model/style_cls_50.pt"))
    print("model loading complete....")

    for param in model.net.conv.parameters(): #learning rate adjust maybe needed
        param.requires_grad=True
    print(list(model.children()))
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    #optimizer

    optimizer = optim.SGD([{'params':model.net.fc.parameters()},{'params':model.net.conv.parameters(), 'lr': 1.e-5},{'params':model.add_fc.parameters(), 'lr': 0.5**(4)*2.5e-3}],lr=0.5**4*2.5e-4,momentum=0.9)#lower the rate for fine-tuning parts
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #data
    transform_train=presets_orig.ClassificationPresetTrain(crop_size=crop_size)
    transform_test=presets_orig.ClassificationPresetEval(crop_size=crop_size)
    wikiart_train=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_train,target_transform=None,split='train')
    wikiart_test=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_test,target_transform=None,split='test')
    train_dataloader = DataLoader(wikiart_train, batch_size=num_batch, shuffle=True)
    test_dataloader = DataLoader(wikiart_test, batch_size=num_batch, shuffle=False)

    #start training
    for t in range(51,num_epochs+1):
        print(f"Epoch {t}\n------------------------------- \n")
        train(train_dataloader, model,criterion,optimizer)
        if t%5==0:
            torch.save(model.state_dict(),"./modeltune_conv/style_cls_"+str(t)+".pt")
        test(test_dataloader, model,criterion)



if __name__=="__main__":
    main()
