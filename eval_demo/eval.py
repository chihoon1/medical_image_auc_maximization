import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

if args.server == "grace":
    os.environ['http_proxy'] = '10.73.132.63:8080'
    os.environ['https_proxy'] = '10.73.132.63:8080'
elif args.server == "faster":
    os.environ['http_proxy'] = '10.72.8.25:8080'
    os.environ['https_proxy'] = '10.72.8.25:8080'

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        if self.trans == None:
            return (self.x[idx], self.y[idx], idx)
        else:
            return (self.trans(self.x[idx]), self.y[idx], idx) 

def train():
    import medmnist 
    from medmnist import INFO, Evaluator
    root = '/scratch/group/optmai/zhishguo/med/'
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:, args.task_index]
    
    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0
    
    # Data Transformation section (Line 49-70). You can change them.
    if test_data.ndim <= 3:
        eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5),
        ])
    elif test_data.ndim == 4:
        eval_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Normalize(0.5, 0.5),
        ])
        test_data = np.swapaxes(test_data, 1, 3) 

    test_data = test_data/255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels) 

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)
    
    from libauc.models import resnet18 as ResNet18
    net = ResNet18(pretrained=False)
    if test_data.ndim == 4:
        # The following line (Line 75) can be changed. 
        net.conv1 = torch.nn.Conv2d(test_data.shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
    net = net.cuda()  
    net.eval() 
    # to save a checkpoint in training: torch.save(net.state_dict(), "saved_model/test_model") 
    if test_data.ndim <= 3:
        #torch.save(net.state_dict(), "saved_model/test_model") 
        net.load_state_dict(torch.load("saved_model/test_model")) 
    if test_data.ndim >= 4:
        #torch.save(net.state_dict(), "saved_model/test_model_3D") 
        net.load_state_dict(torch.load("saved_model/test_model_3D")) 
    evaluate(net, test_loader) 
  
def evaluate(net, test_loader):
    # Testing AUC
    score_list = list()
    label_list = list()
    for _, data in enumerate(test_loader, 0):
        tmp_data, tmp_label, tmp_idx = data
        tmp_data, tmp_label = tmp_data.cuda(), tmp_label.cuda()
                
        tmp_score = net(tmp_data).detach().clone().cpu()
        score_list.append(tmp_score)
        label_list.append(tmp_label.cpu())
    test_label = torch.cat(label_list)
    test_score = torch.cat(score_list)
                   
    test_auc = metrics.roc_auc_score(test_label, test_score)                   
    print("Test: %.4f"%test_auc, flush=True)
     
if __name__ == "__main__":
    train()
