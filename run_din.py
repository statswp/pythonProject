import pandas as pd
from data.amazon_dataloader import get_dataloader
from models import DIN, DCN
import torch

if __name__ == '__main__':
    log_dir = '/Users/wp/PycharmProjects/pythonProject/log_dir'
    train_dataloader, val_dataloader, feature_columns = get_dataloader(batch_size=3)
    for f in feature_columns:
        print(f)

    model = DIN(feature_columns=feature_columns, deep_units=(128, 64, 32), use_bn=True, atten_embedding_dim=16)
    # print(model)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()

    model.compile(optimizer=optimizer, loss_fn=loss_fn)
    model.fit(train_dataloader,
              validate_dataloader=val_dataloader,
              epochs=100,
              log_dir=log_dir,
              log_interval=100)
