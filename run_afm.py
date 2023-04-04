import torch.utils.data

from data.dataloader import get_dataloader
from models import AFM

if __name__ == '__main__':
    log_dir = '/Users/wp/PycharmProjects/pythonProject/log_dir'
    train_dataloader, val_dataloader, feature_columns = get_dataloader()

    model = AFM(feature_columns=feature_columns,
                deep_units=(128, 64, 32),
                embed_size=8,
                atten_factor=4,
                use_bn=True)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()

    model.compile(optimizer=optimizer, loss_fn=loss_fn)
    model.fit(train_dataloader,
              validate_dataloader=val_dataloader,
              epochs=100,
              log_dir=log_dir,
              log_interval=100)
