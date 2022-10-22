import numpy as np
import torch
import torch as t
from torch.utils.data import DataLoader, Dataset
from torch.nn import *
import logging
import argparse
from tqdm import *
global logger


class TensorDataset(Dataset):

    def __init__(self, sparseTensor, offset):
        self.sparseTensor = sparseTensor
        self.offset = offset
        self.tIdx, self.rIdx, self.cIdx = self.sparseTensor.nonzero()

    def __len__(self):
        return len(self.tIdx)

    def __getitem__(self, id):
        tIdx = self.tIdx[id]
        rIdx = self.rIdx[id]
        cIdx = self.cIdx[id]
        mVal = self.sparseTensor[tIdx, rIdx, cIdx]
        return tIdx + self.offset, rIdx, cIdx, mVal


def Metrics(pred, true):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


class CoSTCo(Module):

    def __init__(self, num_times, num_users, num_items, num_shapes, rank):
        super(CoSTCo, self).__init__()
        self.num_shapes = num_shapes
        self.num_channels = rank
        self.time_embeds = Embedding(num_times, rank)
        self.user_embeds = Embedding(num_users, rank)
        self.item_embeds = Embedding(num_items, rank)
        self.conv1 = Sequential(LazyConv2d(self.num_channels, kernel_size=(self.num_shapes, 1)), ReLU())
        self.conv2 = Sequential(LazyConv2d(self.num_channels, kernel_size=(1, rank)), ReLU())
        self.flatten = Flatten()
        self.linear = Sequential(LazyLinear(rank), ReLU())
        self.output = Sequential(LazyLinear(1), ReLU())

    def forward(self, tIdx, rIdx, cIdx):

        # read embeds [batch, dim]
        time_embeds = self.time_embeds(tIdx)
        user_embeds = self.user_embeds(rIdx)
        item_embeds = self.item_embeds(cIdx)

        # stack as [batch, N, dim]
        x = t.stack([time_embeds, user_embeds, item_embeds], dim=1)

        # reshape to [batch, 1, N, dim]
        x = t.unsqueeze(x, dim=1)

        # conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x.flatten()


def get_dataloader(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('../dataset/abilene.npy')[:4000].astype('float32')

    if args.dataset == 'geant':
        tensor = np.load('../dataset/geant.npy')[:3500].astype('float32')

    if args.dataset == 'seattle':
        tensor = np.load('../dataset/seattle.npy')[:650].astype('float32')

    if args.dataset == 'harvard':
        tensor = np.load('../dataset/havard226.npy')[:800].astype('float32')

    if args.dataset == 'taxi':
        tensor = np.load('../dataset/taxi.npy')[:1464].astype('float32')

    thsh = np.percentile(tensor, q=args.quantile)
    tensor[tensor > thsh] = thsh
    tensor /= thsh

    tIdx, srcIdx, dstIdx = tensor.nonzero()
    p = np.random.permutation(len(tIdx))
    tIdx, srcIdx, dstIdx = tIdx[p], srcIdx[p], dstIdx[p]
    sample = int(np.prod(tensor.shape) * args.density)
    stIdx = tIdx[:sample]
    ssrcIdx = srcIdx[:sample]
    sdstIdx = dstIdx[:sample]
    trainTensor = np.zeros_like(tensor)
    trainTensor[stIdx, ssrcIdx, sdstIdx] = tensor[stIdx, ssrcIdx, sdstIdx]

    testTensor = np.zeros_like(tensor)
    stIdx = tIdx[sample:]
    ssrcIdx = srcIdx[sample:]
    sdstIdx = dstIdx[sample:]
    testTensor[stIdx, ssrcIdx, sdstIdx] = tensor[stIdx, ssrcIdx, sdstIdx]

    offset = 0
    if args.dataset == 'abilene':
        testTensor = testTensor[3000:4000]
        offset = 3000

    if args.dataset == 'geant':
        testTensor = testTensor[3000:3500]
        offset = 3000

    if args.dataset == 'seattle':
        testTensor = testTensor[400:650]
        offset = 400

    if args.dataset == 'harvard':
        testTensor = testTensor[600:800]
        offset = 600

    if args.dataset == 'taxi':
        testTensor = testTensor[960:1464]
        offset = 960

    trainset = TensorDataset(trainTensor, offset=0)
    trainLoader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=1)

    testset = TensorDataset(testTensor, offset=offset)
    testLoader = DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=1)

    shape = tensor.shape
    return trainLoader, testLoader, thsh, shape


def run(runid, args):
    trainLoader, testLoader, thsh, shape = get_dataloader(args)
    num_times, num_users, num_items = shape
    model = CoSTCo(num_times, num_users, num_items, num_shapes=3, rank=args.rank).to(args.device)
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
    LossFunc = MSELoss()

    for epoch in trange(args.iter):
        model.train()
        losses = []
        for trainBatch in trainLoader:
            optimizer.zero_grad()
            tIdx, rIdx, cIdx, label = trainBatch
            pred = model.forward(tIdx.to(args.device), rIdx.to(args.device), cIdx.to(args.device))
            loss = LossFunc(pred, label.float().to(args.device))
            loss.backward()
            optimizer.step()
            losses += [float(loss)]

    ERs = []
    NMAEs = []
    model.eval()
    with torch.no_grad():
        for testBatch in testLoader:
            tIdx, rIdx, cIdx, label = testBatch
            pred = model.forward(tIdx.to(args.device), rIdx.to(args.device), cIdx.to(args.device))
            pred = pred.cpu().numpy()
            label = label.numpy()
            ER, NMAE = Metrics(pred * thsh, label * thsh)
            ERs += [ER]
            NMAEs += [NMAE]

    ER = np.mean(ERs)
    NMAE = np.mean(NMAEs)
    logger.info(f"Run ID={runid}, ER={ER:.3f}, NMAE={NMAE:.3f}")
    return ER, NMAE


def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ER, NMAE = run(runid, args)
        RunERs += [ER]
        RunNMAEs += [NMAE]
    logger.info(f'Run ER={np.mean(RunERs):.3f}, Run NAME={np.mean(RunNMAEs):.3f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--density', type=float)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--iter', type=int, default=50)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--quantile', type=int, default=99)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=f'results/CoSTCo_{args.dataset}_{args.density}.log', filemode='w')
    logger = logging.getLogger('CoSTCo')
    logger.info(f'Experiment Config = {args}')
    main(args)





