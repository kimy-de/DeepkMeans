## Deep k-Means (Pytorch)
This code is the prototype of a unsupervised learning model consisting of a convolutional autoencoder and k-Means. To check the implementation, the MNIST dataset is used.

##  Hyperparameters
```
parser.add_argument('--dataset', default='mnist', type=str, help='datasets')
parser.add_argument('--mode', default='train', type=str, help='train or eval')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')   
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')  
parser.add_argument('--num_clusters', default=10, type=int, help='num of clusters') 
parser.add_argument('--latent_size', default=10, type=int, help='size of latent vector') 
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lam', default=1e-2, type=float, help='final rate of clustering loss')
parser.add_argument('--anls', default=10, type=int, help='annealing start point of lambda')
parser.add_argument('--anle', default=110, type=int, help='annealing end point of lambda')
parser.add_argument('--pret', default=None, type=str, help='pretrained model path')
```
## Train
```bash
$ python main.py --mode train --latent_size 10 --epochs 300 --anle 50 --lam 0.005                                    
```

## Evaluation
```bash
$ python main.py --mode eval --latent_size 10 --pret './_'
```

## Results
batch_size = 128,
num_clusters = 10,
latent_size = 10,
T1 = 50,
T2 = 200,
lam = 1e-3,
ls = 0.05

Test accuracy: 89.4%

<img width="400" alt="tsne" src="https://user-images.githubusercontent.com/52735725/142823444-c210738f-f26b-4fdd-98d0-3b9b2ad164b5.png">
<img width="400" alt="스크린샷 2021-11-21 00 05 38" src="https://user-images.githubusercontent.com/52735725/142743644-83f81faa-b478-41da-befc-b2c0391d2809.png">

## To do
- [ ] Revision of the model
- [ ] Comparison among optimization methods
- [ ] Test for other datasets

## Reference
Moradi Fard, M., Thonet, T., & Gaussier, E. (2018) "Deep k-Means: Jointly Clustering with k-Means and Learning Representations", ArXiv:1806.10069.
