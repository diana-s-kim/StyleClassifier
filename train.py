from data.wiki import WikiArt
from stylenet import StyleNet
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import presets_orig

#enable ddp
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

##parameters
num_style=20
crop_size=224
learning_rate=0.001
num_epochs=50
num_batch=64
wiki_csv="./data/wiki.csv"
img_dir="/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/"

##model
netname='vgg16' #original conv - [4096, 10000]
mlp=[[4096,1024],[1024,512],[512,20]]
dropout=[None,None,None]
activations=['relu','relu','relu']


##device 
rank = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")



##loss softmax

def train(dataloader, model, loss_fn, optimizer, rank):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y,_) in enumerate(dataloader):
        X, y = X.to(rank), y.to(rank)

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

def test(dataloader, model, loss_fn, rank):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y,_) in enumerate(dataloader):
            X, y = X.to(rank), y.to(rank)
            pred = model(X)
            current = (batch + 1) * len(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f"{current:>5d}/{size:>5d}")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print("ddpm_setup_yet")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print("ddp_setup_complete")

def ddp_cleanup():
     dist.destroy_process_group()
    
def main(rank,world_size):
    ddp_setup(rank, world_size)
    #build model
    model=StyleNet(name=netname,mlp=mlp,dropout=dropout,activations=activations).to(rank)
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    print("model set up")
    for param in ddp_model.module.net.conv.parameters(): #learning rate adjust maybe needed
        param.requires_grad = False
    print(list(ddp_model.children()))
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    #optimizer
#    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.SGD([{'params':ddp_model.module.net.parameters()},{'params':ddp_model.module.add_fc.parameters(), 'lr': 2.5e-3}],lr=2.5e-4,momentum=0.9)#lower the rate for fine-tuning parts
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #data
    transform_train=presets_orig.ClassificationPresetTrain(crop_size=crop_size)
    transform_test=presets_orig.ClassificationPresetEval(crop_size=crop_size)
    wikiart_train=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_train,target_transform=None,split='train')
    wikiart_test=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_test,target_transform=None,split='test')
    train_dataloader = DataLoader(wikiart_train, batch_size=num_batch, shuffle=False,sampler=DistributedSampler(wikiart_train))
    test_dataloader = DataLoader(wikiart_train, batch_size=num_batch, shuffle=False,sampler=DistributedSampler(wikiart_test))

    #start training
    for t in range(1,num_epochs+1):
        print(f"Epoch {t}\n------------------------------- \n")
        train_dataloader.sampler.set_epoch(t)
        train(train_dataloader,ddp_model,criterion,optimizer,rank)
        if t%5==0:
            torch.save(ddp_model.module.state_dict(),"./model/style_cls_"+str(t)+".pt")
        test_dataloader.sampler.set_epoch(t)
        test(test_dataloader,ddp_model,criterion,rank)
    clearn_up()
        

if __name__=="__main__":
    world_size = torch.cuda.device_count()    
    mp.spawn(main, args=(world_size,), nprocs=world_size,join=True)
