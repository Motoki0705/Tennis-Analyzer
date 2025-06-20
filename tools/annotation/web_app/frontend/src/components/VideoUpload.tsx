import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Divider,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';
import axios from 'axios';

interface VideoMetadata {
  filename: string;
  file_path: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  file_size: number;
  upload_date: string;
}

const VideoUpload: React.FC = () => {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const fetchVideos = useCallback(async () => {
    try {
      const response = await axios.get('/api/raw_videos');
      setVideos(response.data);
    } catch (err) {
      console.error('動画リスト取得エラー:', err);
      setError('動画リストの取得に失敗しました');
    }
  }, []);

  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    // ファイルタイプチェック
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
    if (!allowedTypes.includes(file.type)) {
      setError('サポートされていないファイル形式です。MP4, AVI, MOV, MKVファイルをアップロードしてください。');
      return;
    }

    // ファイルサイズチェック（10GB制限）
    const maxSize = 10 * 1024 * 1024 * 1024; // 10GB
    if (file.size > maxSize) {
      setError('ファイルサイズが10GBを超えています。');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append('video_file', file);

    try {
      const response = await axios.post('/api/upload_video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(progress);
          }
        },
      });

      setSuccess(`動画アップロード完了: ${response.data.metadata.filename}`);
      await fetchVideos(); // リストを更新
    } catch (err: any) {
      console.error('アップロードエラー:', err);
      setError(err.response?.data?.detail || 'アップロードに失敗しました');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        動画アップロード
      </Typography>

      {/* アップロードエリア */}
      <Card 
        sx={{ 
          mb: 3,
          border: dragOver ? '2px dashed #1976d2' : '2px dashed #ccc',
          backgroundColor: dragOver ? 'rgba(25, 118, 210, 0.1)' : 'inherit',
        }}
      >
        <CardContent>
          <Box
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            sx={{
              textAlign: 'center',
              py: 4,
              cursor: 'pointer',
            }}
          >
            <CloudUploadIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              動画ファイルをドラッグ&ドロップ
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              または
            </Typography>
            <input
              accept="video/*"
              style={{ display: 'none' }}
              id="video-upload-button"
              type="file"
              onChange={handleFileSelect}
              disabled={uploading}
            />
            <label htmlFor="video-upload-button">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUploadIcon />}
                disabled={uploading}
              >
                ファイルを選択
              </Button>
            </label>
            <Typography variant="caption" display="block" sx={{ mt: 2 }}>
              対応形式: MP4, AVI, MOV, MKV（最大10GB）
            </Typography>
          </Box>

          {uploading && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                アップロード中... {uploadProgress}%
              </Typography>
              <LinearProgress variant="determinate" value={uploadProgress} />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* エラー・成功メッセージ */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* アップロード済み動画リスト */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            アップロード済み動画 ({videos.length}件)
          </Typography>
          
          {videos.length === 0 ? (
            <Typography color="text.secondary">
              アップロードされた動画はありません
            </Typography>
          ) : (
            <List>
              {videos.map((video, index) => (
                <React.Fragment key={video.filename}>
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle1">
                            {video.filename}
                          </Typography>
                          <Chip 
                            label={`${video.width}x${video.height}`} 
                            size="small" 
                            variant="outlined" 
                          />
                          <Chip 
                            label={`${video.fps.toFixed(1)} FPS`} 
                            size="small" 
                            variant="outlined" 
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2">
                            長さ: {formatDuration(video.duration)} | 
                            サイズ: {formatFileSize(video.file_size)} | 
                            フレーム数: {video.total_frames.toLocaleString()}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            アップロード日時: {new Date(video.upload_date).toLocaleString('ja-JP')}
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton edge="end" aria-label="プレビュー">
                        <PlayArrowIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < videos.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default VideoUpload; 