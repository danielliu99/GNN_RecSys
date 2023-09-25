import os
import argparse
import torch
import numpy as np
from tqdm.notebook import tqdm

from model.GNNRecSys import GNNRecSys
from data.dataset import data_loader
from data.preprocess import feature_encoding
from utils.metrics import compute_bpr_loss, get_metrics

LATENT_DIM = 4
N_LAYERS = 1

EPOCHS = 1
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005
K = 20


def train_and_eval(model, optimizer, train_df, train_edge_index):
    loss_list_epoch = []
    bpr_loss_list_epoch = []
    reg_loss_list_epoch = []

    recall_list = []
    precision_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_edge_index = train_edge_index.to(device)
    for epoch in tqdm(range(EPOCHS)):
        n_batch = int(len(train_df) / BATCH_SIZE)

        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []

        model.train()
        for batch_idx in tqdm(range(n_batch)):
            print(batch_idx)
            optimizer.zero_grad()

            users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items, device)
            users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = model.encode_minibatch(users, pos_items,
                                                                                             neg_items,
                                                                                             train_edge_index)

            bpr_loss, reg_loss = compute_bpr_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0
            )
            reg_loss = DECAY * reg_loss
            final_loss = bpr_loss + reg_loss
            print(final_loss)

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            test_topK_recall, test_topK_precision = get_metrics(
                final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K
            )

        loss_list_epoch.append(round(np.mean(final_loss_list), 4))
        bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list), 4))
        reg_loss_list_epoch.append(round(np.mean(reg_loss_list), 4))

        recall_list.append(round(test_topK_recall, 4))
        precision_list.append(round(test_topK_precision, 4))

    print("Finished epochs")
    return (
        loss_list_epoch,
        bpr_loss_list_epoch,
        reg_loss_list_epoch,
        recall_list,
        precision_list,
        final_user_Embed,
        final_item_Embed
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str)
    parser.add_argument('-m', '--model', default="LightGCN", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_df, test_df, n_users, n_items, u_t, i_t, train_edge_index = feature_encoding(
        train_file_path=os.path.join(args.data_dir, "train.csv"),
        test_file_path=os.path.join(args.data_dir, "test.csv")
    )

    if args.model == 'LightGCN':
        model = GNNRecSys(
            latent_dim=LATENT_DIM,
            num_layers=N_LAYERS,
            num_users=n_users,
            num_items=n_items,
            model='LightGCN'
        )
    else:
        raise Exception("Not an accepted model")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Size of Learnable Embedding : ", [x.shape for x in list(model.parameters())])

    light_loss, light_bpr, light_reg, light_recall, light_precision, user_emb, item_emb = train_and_eval(model,
                                                                                                         optimizer,
                                                                                                         train_df,
                                                                                                         train_edge_index)
    ## save model weights
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    torch.save(user_emb, "weights/user_embedding.pt")
    torch.save(item_emb, "weights/item_embedding.pt")
    print("Weights saved.")
    if not os.path.isdir("out"):
        os.mkdir("out")
    torch.save(light_bpr, "out/bpr.pt")
    torch.save(light_reg, "out/reg.pt")
    