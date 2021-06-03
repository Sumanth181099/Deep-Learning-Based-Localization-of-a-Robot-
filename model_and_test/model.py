import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
from dataloading import ImageDataset
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

# hyperparameters
batch_size = 32
num_epochs = 200  
learning_rate = 0.001
weight_decay = 0.0005
# input_size = 1024
hidden_size = 256
num_layers = 2
# seq_len = 1
out_pose = 3
beta = 1
patience = 15
# clip_value = 100
# input to lstm  is 1024 sized vector = 32*32
input_size = 32
seq_len = 32
# setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/3D_localization_2seq_shuffled_dataset_2')
# seeding
torch.manual_seed(0)


def init_weights(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.zeros_(param)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
    return module


def print_param(model):
    for name, param in model.named_parameters():
        print('name: ', name)
#        if name in ['resnet.fc.weight', 'resnet.fc.bias']:
#            param.requires_grad = True
        # print(type(param))
        # print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')
# Model


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_pose):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_pose = out_pose
        resnet = torchvision.models.resnet50(pretrained=True)
        # modules = list(resnet.children())[:]
        # print(modules)
        in_feat = resnet.fc.in_features
        resnet.fc = nn.Linear(in_feat, 1024)
        self.resnet = resnet
        # self.resnet = nn.Sequential(*modules)
        for name, param in self.resnet.named_parameters():
            # print('name: ', name)
            if name in ['fc.weight', 'fc.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # self.fc0 = nn.Linear(in_features = 2048, out_features = input_size)
        self.rnn = init_weights(nn.LSTM(input_size, hidden_size, num_layers,
                                        bias=True, batch_first=True,
                                        bidirectional=True, dropout=0.3))
        # self.fc1 = nn.Linear(hidden_size*seq_len, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2*hidden_size, 128)
        self.fc2 = nn.Linear(128, out_pose)

    def forward(self, x):
        h0 = torch.randn(2*self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.randn(2*self.num_layers, x.size(0), self.hidden_size).cuda()
        # batch_size, c,h,w = x.size()
        conv_out = self.resnet(x)
        # conv_out = self.fc0(resnet_out)
        # print("Output size of conv_layer is", conv_out.size())
        lstm_in = conv_out.view(batch_size, seq_len, -1)
        lstm_out, (hidden, cell) = self.rnn(lstm_in, (h0, c0))
        f_out = torch.cat((hidden[0, :, :],
                          hidden[1, :, :]), 1)
        # elu = nn.ELU()
        # f_out = elu(f_out)
        f_out = self.fc1(f_out)
        # f_out = self.dropout(f_out)
        f_out = self.fc2(f_out)
        return f_out

# initialize the model


model = CNN_LSTM(input_size, hidden_size, num_layers, out_pose)
model.to(device)
# Loading the data compatible with resnet50
image_dataset = ImageDataset(root_dir='Overall_dataset_2_seqs/overall_img_frames',
                             csv_file="new_shuffled_dataset.csv",
                             transform=transforms.Compose(
                                                        [transforms.ToPILImage(),
                                                         transforms.Resize(256),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                         [0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])

train_set = Subset(image_dataset, indices=range(2016))
valid_set = Subset(image_dataset, indices=range(2016, 2080))
test_set = Subset(image_dataset, indices=range(2080, len(image_dataset)))
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                         shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       factor=0.1, patience=7,
                                                       verbose=True)

# loss function - L1 loss

criterion = nn.L1Loss()

# training function


def training(model, criterion, train_dataloader=train_loader,
             optimizer=optimizer, epochs=num_epochs):
    print("Training")
    train_loss = []
    valid_loss = []
    # model.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        valid_running_loss = 0.0
        for batch_idx, (img, target) in enumerate(train_loader):
            # forward pass
            img = img.to(device=device)
            target = target.to(device=device)
            optimizer.zero_grad()
            output = model(img)
            # loss and backward pass
            loss = criterion(output, target)
            # print(loss)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            running_loss += loss.item()
            # validation loss calculation
        model.eval()
        for batch, (img, target) in enumerate(test_loader):
            img = img.to(device=device)
            target = target.to(device=device)
            output = model(img)
            loss = criterion(output, target)
            valid_running_loss += loss.item()
        v_loss = valid_running_loss/len(test_loader)
        loss = running_loss/len(train_loader)
        scheduler.step(loss)
        train_loss.append(loss)
        valid_loss.append(v_loss)
        early_stopping(v_loss, model)
        print("Epoch {} of {}, Train_loss: {:.4f}, V_Loss: {:.4f}" .format(
              epoch+1, epochs, loss, v_loss))
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('validation_loss', v_loss, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # torch.save(model.state_dict(), '3D_visual_localization_28/11.pt')
        # model.load_state_dict(torch.load('3D_vloc_28_11.pt'))
    print("Done with training!")
    return train_loss, valid_loss


def accuracy_check(data, model):
    if data == train_loader:
        print("Training accuracy being calculated")
    else:
        print("Accuracy of test data frames")
    # num_correct = 0
    num_correct_easy = 0
    num_samples = 0
    # err_margin = torch.tensor([[0.1, 0.1, 10]])
    err_margin_easy = torch.tensor([[0.5, 0.5, 10]])
    model.load_state_dict(torch.load('3D_visual_loc_2_12.pt'))
    model.eval()
    error = []
    out_reg = []
    g_t = []
    with torch.no_grad():
        for i, (img, target) in enumerate(data):
            img = img.to(device=device)
            target = target.to(device=device)
            output = model(img)
            out_tar = abs(output - target)
            three_D_ground_truth = target.cpu().numpy()
            err_scalar = out_tar.cpu().numpy()
            out_xy = output.cpu().numpy()
            for j in err_scalar:
                error.append(j)
            for k in out_xy:
                out_reg.append(k)
            for t in three_D_ground_truth:
                g_t.append(t)
            np.savetxt("3Dtest_gt_2_12.csv", g_t, delimiter=",", fmt='% f')
            np.savetxt("3Dtest_error_2_12.csv", error, delimiter=",", fmt='% f')
            np.savetxt("3Dtest_regout_2_12.csv", out_reg, delimiter=",", fmt='% f')
            for x, y, phi in out_tar:
                if((x < err_margin_easy[0][0]) and (y < err_margin_easy[0][1])
                   and (phi < err_margin_easy[0][2])):
                    num_correct_easy += 1
            num_samples += output.size(0)
            accuracy_easy = (100*num_correct_easy)/num_samples
            # v_loss = run_loss/len(data)
            print("Got {}/{} with easy_accuracy: {:.2f}".format(
                   num_correct_easy, num_samples, accuracy_easy))
            # valid_loss.append(v_loss)
            # writer.add_scalar('validation_loss', v_loss, len(valid_loss))
    # model.train()
    return accuracy_easy


def main():
    # training(model=model, criterion=criterion, train_dataloader=train_loader,
    #         optimizer=optimizer, epochs=num_epochs)
    accuracy_check(data=test_loader, model=model)


if __name__ == "__main__":
    main()
