import torch
import torch.nn as nn

from torch.autograd import Variable
import argparse
import torch.optim as optim

import networks_adaCBMA_deform as networks
from data import Gopro
from utils import *
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from apex import amp


parser = argparse.ArgumentParser(description='image-deblurring')

# train data
parser.add_argument('--data_dir', type=str, required=True,
                    help='dataset directory')  # modifying to your training data folder path
parser.add_argument('--save_dir', default='./result', help='data save directory')
parser.add_argument('--patch_size', type=int, default=256, help='patch size')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')

# validation data
parser.add_argument('--val_data_dir', type=str, default='F:/Dataset/GOPRO_Large/val')  # modifying to validation data folder path
#parser.add_argument('--val_data_dir', type=str, default=None)
parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')

# model
parser.add_argument('--exp_name', default='AdaCAMB_skip_Deform1x1', help='model to select')
parser.add_argument('--finetuning', action='store_true', help='finetuning the training')

# network
parser.add_argument('--multi', action='store_true', help='Using long skip connection')
parser.add_argument('--n_resblocks', type=int, default=9, help='number of residual block')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--adaNorm', type=bool, default=False, help='whether to use adaptive LINorm or not')
parser.add_argument('--deformable', type=bool, default=True, help='using deformable convnet')
parser.add_argument('--deformable_kersize', type=int, default=1, help='deformable_kersize')
# optimization
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
parser.add_argument('--lr_step_size', type=int, default=1000,  help='period of learning rate decay')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')

#
parser.add_argument('--period', type=int, default=100, help='period of evaluation')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)



def get_dataset(data_dir, patch_size=None, batch_size=1, n_threads=8, is_train=False, multi=False):
    dataset = Gopro(data_dir, patch_size=patch_size, is_train=is_train, multi=multi)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=is_train, num_workers=int(n_threads))
    return dataloader

 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validation(model, dataloader, multi):
    total_psnr = 0
    for batch, images in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            output_l1 = model(input_b1)

        output_l1 = tensor_to_rgb(output_l1)
        target_s1 = tensor_to_rgb(target_s1)

        psnr = compute_psnr(target_s1, output_l1)
        total_psnr += psnr

    return total_psnr / (batch + 1)


def train(args):
    print(args)

    my_model = networks.ResnetGenerator(3,3,ngf=args.n_feats,n_blocks=args.n_resblocks,
                                        adaNorm=args.adaNorm,deformable=args.deformable,dcn_ksize=args.deformable_kersize)
    my_model = my_model.cuda()
   
    loss_function = nn.MSELoss().cuda()
    optimizer = optim.Adam(my_model.parameters(), lr=args.lr)
    

    # utility for saving models, parameters and logs
    save = SaveData(args.save_dir, args.exp_name, args.finetuning)
    save.save_params(args)
    num_params = count_parameters(my_model)
    save.save_log(str(num_params))

   
   # using apex
    my_model, optimizer = amp.initialize(my_model, optimizer, opt_level="O1")

   
    # load pre-trained model if provided
    last_epoch = -1
    if args.finetuning:
        my_model, last_epoch, optimizer = save.load_model(my_model,optimizer)
        #optimizer.param_groups[0]['lr']=1e-5
    start_epoch = last_epoch + 1

    

    scheduler = lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma,last_epoch=last_epoch)

    # load dataset
    data_loader = get_dataset(args.data_dir, patch_size=args.patch_size, batch_size=args.batch_size,
                              n_threads=args.n_threads, is_train=True, multi=args.multi)
    if args.val_data_dir:
        valid_data_loader = get_dataset(args.val_data_dir, n_threads=args.n_threads, multi=args.multi)

    
    for epoch in range(start_epoch, args.epochs):
        print("* Epoch {}/{}".format(epoch + 1, args.epochs))
        learning_rate = optimizer.param_groups[0]['lr']
        total_loss = 0

        for batch, images in tqdm(enumerate(data_loader)):
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            output_l1 = my_model(input_b1)
            loss = loss_function(output_l1, target_s1)

            my_model.zero_grad()
            
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()


            optimizer.step()
            total_loss += loss.data.cpu().numpy()

        loss = total_loss / (batch + 1)
        save.add_scalar('train/loss', loss, epoch)
        
        #wandb.log({'epoch': epoch, 'loss': loss})

        if  epoch % args.period == 0:

            if args.val_data_dir:
                torch.cuda.empty_cache()
                my_model.eval()
                psnr = validation(my_model, valid_data_loader, args.multi)
                my_model.train()

                log = "Epoch {}/{} \t Learning rate: {:.10f} \t Train total_loss: {:.5f} \t * Val PSNR: {:.2f}\n".format(
                    epoch + 1, args.epochs, learning_rate, loss, psnr)
                print(log)
                save.save_log(log)
                save.add_scalar('valid/psnr', psnr, epoch)
        else:
            log = "Epoch {}/{} \t Learning rate: {:.10f} \t Train total_loss: {:.5f}\n".format(epoch + 1, args.epochs,
                                                                                              learning_rate, loss)
            print(log)
            save.save_log(log)

        save.save_model(my_model, epoch,optimizer)
        scheduler.step()


if __name__ == '__main__':
    train(args)
