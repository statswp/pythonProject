import torch.utils.data

from data.dataloader import get_dataloader
from models import DeepFM

if __name__ == '__main__':
    log_dir = '/Users/wp/PycharmProjects/pythonProject/log_dir'
    train_dataloader, val_dataloader, feature_columns = get_dataloader()

    # DeepFM
    deepfm_model = DeepFM(feature_columns=feature_columns, deep_units=(128, 64, 32), use_bn=True)
    optimizer = torch.optim.SGD(params=deepfm_model.parameters(), lr=0.0001)
    loss_fn = torch.nn.BCELoss()

    deepfm_model.compile(optimizer=optimizer, loss_fn=loss_fn)
    deepfm_model.fit(train_dataloader,
                     validate_dataloader=val_dataloader,
                     epochs=100,
                     log_dir=log_dir,
                     log_interval=100)
