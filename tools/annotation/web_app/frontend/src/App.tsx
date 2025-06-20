import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
} from '@mui/material';
import {
  VideoLibrary as VideoLibraryIcon,
  ContentCut as ContentCutIcon,
  Assignment as AssignmentIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';

// Import components (will be created next)
import VideoUpload from './components/VideoUpload';
import ClipCreation from './components/ClipCreation';
import AnnotationList from './components/AnnotationList';
import AnnotationEditor from './components/AnnotationEditor';
import Statistics from './components/Statistics';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const drawerWidth = 240;

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <AppBar
            position="fixed"
            sx={{ width: `calc(100% - ${drawerWidth}px)`, ml: `${drawerWidth}px` }}
          >
            <Toolbar>
              <Typography variant="h6" noWrap component="div">
                テニスイベントアノテーションツール
              </Typography>
            </Toolbar>
          </AppBar>

          <Drawer
            sx={{
              width: drawerWidth,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: drawerWidth,
                boxSizing: 'border-box',
              },
            }}
            variant="permanent"
            anchor="left"
          >
            <Toolbar />
            <List>
              <ListItem button component={Link} to="/upload">
                <ListItemIcon>
                  <VideoLibraryIcon />
                </ListItemIcon>
                <ListItemText primary="動画アップロード" />
              </ListItem>

              <ListItem button component={Link} to="/clip-creation">
                <ListItemIcon>
                  <ContentCutIcon />
                </ListItemIcon>
                <ListItemText primary="クリップ生成" />
              </ListItem>

              <ListItem button component={Link} to="/annotation">
                <ListItemIcon>
                  <AssignmentIcon />
                </ListItemIcon>
                <ListItemText primary="アノテーション" />
              </ListItem>

              <ListItem button component={Link} to="/statistics">
                <ListItemIcon>
                  <AssessmentIcon />
                </ListItemIcon>
                <ListItemText primary="統計" />
              </ListItem>
            </List>
          </Drawer>

          <Box
            component="main"
            sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}
          >
            <Toolbar />
            <Container maxWidth="lg">
              <Routes>
                <Route path="/" element={<Navigate to="/upload" replace />} />
                <Route path="/upload" element={<VideoUpload />} />
                <Route path="/clip-creation" element={<ClipCreation />} />
                <Route path="/annotation" element={<AnnotationList />} />
                <Route path="/annotation/:clipName" element={<AnnotationEditor />} />
                <Route path="/statistics" element={<Statistics />} />
              </Routes>
            </Container>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App; 