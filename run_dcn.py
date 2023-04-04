import torch.utils.data

from data import dataloader
from models import DCN

if __name__ == '__main__':
    log_dir = '/Users/wp/PycharmProjects/pythonProject/log_dir'
    train_dataloader, val_dataloader, feature_columns = dataloader.get_dataloader(embed_dim=8)
    # train_dataloader, val_dataloader, feature_columns = dataloader.get_dataloader_v2(embed_dim=8)
    for f in feature_columns:
        print(f)

    dcn_model = DCN(feature_columns=feature_columns, deep_units=(128, 64, 32), use_bn=True)
    optimizer = torch.optim.SGD(params=dcn_model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()

    dcn_model.compile(optimizer=optimizer, loss_fn=loss_fn)
    dcn_model.fit(train_dataloader,
                  validate_dataloader=val_dataloader,
                  epochs=100,
                  log_dir=log_dir,
                  log_interval=100)
