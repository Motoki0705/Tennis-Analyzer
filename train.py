import torch
import torch.nn as nn
from create_dataset import FrameDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import model
from config import Config
    
def train(
    encoder,
    decoder,
    criterion,
    optimizer_encoder,
    optimizer_decoder,
    train_loader,
    device,
    config: Config
    ):
    encoder.train()
    decoder.train()

    best_loss = float('inf')  # 最良モデルの検証損失を初期化
    best_encoder_path = config.best_encoder_path
    best_decoder_path = config.best_decoder_path

    for epoch in range(config.epochs):
        hidden_state = None
        epoch_loss = 0.0  # エポック全体の損失を初期化
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}", total=len(train_loader))

        # トレーニングループ
        for idx, (frames, label_state, label_event) in progress_bar:
            frames, label_state, label_event = frames.to(device), label_state.to(device), label_event.to(device)
            if frames.size(0) == 1:
                continue

            # モデルの順伝播
            encoder_out, x_resolution = encoder(frames)
            x_resolution = torch.unsqueeze(x_resolution[-1], dim=0)
            decoder_out_state = []
            decoder_out_event = []
            for i in range(encoder_out.size(0)):
                output_state, hidden_state = decoder(
                    encoder_out[i], 
                    x_resolution[:, i],
                    [h.detach() for h in hidden_state] if hidden_state is not None else None,
                    is_state = True
                )
                output_event, hidden_state = decoder(
                    encoder_out[i], 
                    x_resolution[:, i],
                    [h.detach() for h in hidden_state] if hidden_state is not None else None,
                    is_event = True
                )
 
                decoder_out_state.append(output_state)
                decoder_out_event.append(output_event)
                
            decoder_out_state = torch.squeeze(torch.stack(decoder_out_state, dim=0), dim=1) #(batch, 1, 3) -> (batch, 3)
            decoder_out_event = torch.squeeze(torch.stack(decoder_out_event, dim=0), dim=1) #(batch, 1, 14) -> (batch, 14)

            
            # 勾配初期化
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            # 損失計算
            loss = criterion(decoder_out_state, label_state) + criterion(decoder_out_event, label_event)
            

            # 逆伝播とパラメータ更新
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            # 損失を記録
            epoch_loss += loss.item()

            # TQDMに損失を表示
            progress_bar.set_postfix(loss=loss.item())

        # エポックごとの平均損失を計算
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # 検証損失が最良であればモデルを保存
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), best_encoder_path)
            torch.save(decoder.state_dict(), best_decoder_path)
            print(f"Saved best model with train_loss: {epoch_loss:.4f}")

    print("Training complete!")
            
    return epoch_loss / len(train_loader)
            
if __name__ == '__main__':
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = model.Encoder(num_blocks=config.num_blocks, input_channels=config.input_channels).to(device)
    decoder = model.Decoder(num_blocks=config.num_blocks).to(device)

    # オプティマイザ
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.001)
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)

    # 損失関数
    criterion = nn.MSELoss()

    # データ変換とデータセット
    transformer = transforms.Compose([
        transforms.Resize(config.input_frame_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = FrameDataset(config.frame_directory, transformer, config, train=True)
    train_loader = DataLoader(dataset, batch_size=config.batch_size)

    # 学習
    avg_loss = train(
        encoder,
        decoder,
        criterion,
        optimizer_encoder,
        optimizer_decoder,
        train_loader,
        device,
        config
    )

    print(f"Average Loss: {avg_loss:.4f}")
