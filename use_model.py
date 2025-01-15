from create_dataset import FrameDataset
from config import Config
from model import Encoder, Decoder
import cv2 
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import threading
import queue


class PlayModelPrediction():
    def __init__(self, encoder, decoder, fps, device, config: Config):
        self.delay = 1 / fps
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        
        self.frame_paths = self.prepare_frame_paths()
        
        #キューをセット
        self.prediction_queue = queue.Queue(maxsize=100)
        
        #動画再生関数とモデル関数をthread化
        self.play_video_thread = threading.Thread(target=self.play_video, daemon=True)
        self.prepare_prediction_thread = threading.Thread(target=self.prepare_prediction, daemon=True)
             
    def play_video(self):
        while True:
            frame, state_prediction, event_prediction = self.prediction_queue.get()
            
            if frame is None:
                print('Could not read frame.')
            
            self.visualize_prediction(frame, state_prediction, event_prediction)
            cv2.imshow('Video PlayBack', frame)
            
            if cv2.waitKey(int(self.delay * 1000)) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()        
    
    def visualize_prediction(self, frame, state_prediction, event_prediction):
        state_index =torch.argmax(state_prediction).item()
        event_index = torch.argmax(event_prediction).item()
        
        label_state = f'state : {self.config.states[state_index]}'
        position_state = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        linn_type = cv2.LINE_8
        
        text_size, baseline = cv2.getTextSize(label_state, font, font_scale, thickness)
        text_w, text_h = text_size
        text_x, text_y = position_state
        
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 255, 0), 2)
        cv2.putText(frame, label_state, position_state, font, font_scale, color, thickness, linn_type)
        
        label_event = f'event : {self.config.events[event_index]}'
        position_event = (10, 30 + text_h + 20)
        
        text_size, baseline = cv2.getTextSize(label_event, font, font_scale, thickness)
        text_w, text_h = text_size
        text_x, text_y = position_event
        
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 255, 0), 2)
        cv2.putText(frame, label_event, position_event, font, font_scale, color, thickness, linn_type)
    
    def prepare_prediction(self):
        pred_hidden_state = None
        pred_hidden_event = None
        transformer = transforms.Compose([
            transforms.Resize(self.config.input_frame_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        for frame_path in self.frame_paths:
        
            frame = cv2.imread(frame_path)
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pillow = Image.fromarray(frame_RGB)
            
            frame_transformed = transformer(frame_pillow).to(self.device)
            frame_transformed = torch.unsqueeze(frame_transformed, dim=0)
            
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                enc_out, x_resolutions = self.encoder(frame_transformed)
                enc_out = torch.squeeze(enc_out, dim=0)
                x_resolutions = torch.unsqueeze(x_resolutions[-1], dim=0)
                
                state_prediction, hidden_state = self.decoder(enc_out, x_resolutions, pred_hidden_state, is_state=True)
                event_prediction, hidden_event = self.decoder(enc_out, x_resolutions, pred_hidden_event, is_event=True)
            
            pred_hidden_state = hidden_state
            pred_hidden_event = hidden_event
            
            self.prediction_queue.put((frame, state_prediction, event_prediction))
        
    
    def prepare_frame_paths(self):
        
        dataset = FrameDataset(config.frame_directory, transformer=None)
        frame_paths = dataset.frame_paths
        return frame_paths    
    
    def start_threads(self):
        self.play_video_thread.start()
        self.prepare_prediction_thread.start()
        
    def join_threads(self):
        self.play_video_thread.join()
        self.prepare_prediction_thread.join()
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    encoder = Encoder(num_blocks=config.num_blocks, input_channels=config.input_channels).to(device)
    decoder = Decoder(num_blocks=config.num_blocks).to(device)
    state_dict_encoder = torch.load('best_encoder.pth')
    state_dict_decoder = torch.load('best_decoder.pth')
    encoder.load_state_dict(state_dict_encoder)
    decoder.load_state_dict(state_dict_decoder)
    
    
    video_player = PlayModelPrediction(encoder, decoder, fps=30, device=device, config=config)
    video_player.start_threads()
    video_player.join_threads()