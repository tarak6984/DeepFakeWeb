import type { AnalysisResult } from './local-deepfake-api';

const STORAGE_KEYS = {
  ANALYSIS_HISTORY: 'deepfake_analysis_history',
  THEME: 'deepfake_theme',
  USER_PREFERENCES: 'deepfake_preferences',
} as const;

export interface StoredAnalysis extends AnalysisResult {
  thumbnailBlob?: string; // Base64 encoded thumbnail
}

export interface UserPreferences {
  maxHistoryItems: number;
  showDetailedAnalysis: boolean;
  autoDownloadReports: boolean;
  theme: 'light' | 'dark' | 'system';
}

const DEFAULT_PREFERENCES: UserPreferences = {
  maxHistoryItems: 10,
  showDetailedAnalysis: true,
  autoDownloadReports: false,
  theme: 'system',
};

class StorageManager {
  private isClient = typeof window !== 'undefined';

  // Analysis History Management
  getAnalysisHistory(): StoredAnalysis[] {
    if (!this.isClient) return [];
    
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.ANALYSIS_HISTORY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to load analysis history:', error);
      return [];
    }
  }

  addAnalysisToHistory(analysis: AnalysisResult, thumbnailBlob?: string): void {
    if (!this.isClient) return;

    try {
      const history = this.getAnalysisHistory();
      const preferences = this.getUserPreferences();
      
      const storedAnalysis: StoredAnalysis = {
        ...analysis,
        thumbnailBlob,
      };

      // Add to beginning of array (most recent first)
      const updatedHistory = [storedAnalysis, ...history];
      
      // Limit history size
      const limitedHistory = updatedHistory.slice(0, preferences.maxHistoryItems);
      
      localStorage.setItem(STORAGE_KEYS.ANALYSIS_HISTORY, JSON.stringify(limitedHistory));
    } catch (error) {
      console.error('Failed to save analysis to history:', error);
    }
  }

  removeAnalysisFromHistory(analysisId: string): void {
    if (!this.isClient) return;

    try {
      const history = this.getAnalysisHistory();
      const filteredHistory = history.filter(analysis => analysis.id !== analysisId);
      localStorage.setItem(STORAGE_KEYS.ANALYSIS_HISTORY, JSON.stringify(filteredHistory));
    } catch (error) {
      console.error('Failed to remove analysis from history:', error);
    }
  }

  clearAnalysisHistory(): void {
    if (!this.isClient) return;
    
    try {
      localStorage.removeItem(STORAGE_KEYS.ANALYSIS_HISTORY);
    } catch (error) {
      console.error('Failed to clear analysis history:', error);
    }
  }

  getAnalysisById(analysisId: string): StoredAnalysis | null {
    const history = this.getAnalysisHistory();
    return history.find(analysis => analysis.id === analysisId) || null;
  }

  // User Preferences Management
  getUserPreferences(): UserPreferences {
    if (!this.isClient) return DEFAULT_PREFERENCES;

    try {
      const stored = localStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to ensure all properties exist
        return { ...DEFAULT_PREFERENCES, ...parsed };
      }
    } catch (error) {
      console.error('Failed to load user preferences:', error);
    }
    
    return DEFAULT_PREFERENCES;
  }

  updateUserPreferences(preferences: Partial<UserPreferences>): void {
    if (!this.isClient) return;

    try {
      const current = this.getUserPreferences();
      const updated = { ...current, ...preferences };
      localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(updated));
    } catch (error) {
      console.error('Failed to save user preferences:', error);
    }
  }

  // Theme Management
  getTheme(): 'light' | 'dark' | 'system' {
    if (!this.isClient) return 'system';

    try {
      const stored = localStorage.getItem(STORAGE_KEYS.THEME);
      if (stored && ['light', 'dark', 'system'].includes(stored)) {
        return stored as 'light' | 'dark' | 'system';
      }
    } catch (error) {
      console.error('Failed to load theme preference:', error);
    }
    
    return 'system';
  }

  setTheme(theme: 'light' | 'dark' | 'system'): void {
    if (!this.isClient) return;

    try {
      localStorage.setItem(STORAGE_KEYS.THEME, theme);
      this.updateUserPreferences({ theme });
    } catch (error) {
      console.error('Failed to save theme preference:', error);
    }
  }

  // Data Export/Import
  exportData(): string {
    if (!this.isClient) return '';

    const data = {
      analysisHistory: this.getAnalysisHistory(),
      userPreferences: this.getUserPreferences(),
      theme: this.getTheme(),
      exportDate: new Date().toISOString(),
    };

    return JSON.stringify(data, null, 2);
  }

  importData(jsonData: string): { success: boolean; message: string } {
    if (!this.isClient) {
      return { success: false, message: 'Import not available on server side' };
    }

    try {
      const data = JSON.parse(jsonData);
      
      if (data.analysisHistory) {
        localStorage.setItem(STORAGE_KEYS.ANALYSIS_HISTORY, JSON.stringify(data.analysisHistory));
      }
      
      if (data.userPreferences) {
        localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(data.userPreferences));
      }
      
      if (data.theme) {
        localStorage.setItem(STORAGE_KEYS.THEME, data.theme);
      }

      return { success: true, message: 'Data imported successfully' };
    } catch (error) {
      console.error('Failed to import data:', error);
      return { success: false, message: 'Invalid data format' };
    }
  }

  // Storage Statistics
  getStorageStats(): { 
    historyCount: number; 
    estimatedSize: string;
    lastModified?: string;
  } {
    if (!this.isClient) {
      return { historyCount: 0, estimatedSize: '0 KB' };
    }

    try {
      const history = this.getAnalysisHistory();
      const dataStr = localStorage.getItem(STORAGE_KEYS.ANALYSIS_HISTORY) || '';
      const sizeInBytes = new Blob([dataStr]).size;
      const sizeInKB = Math.round(sizeInBytes / 1024 * 100) / 100;
      
      const lastModified = history.length > 0 
        ? history[0].timestamp 
        : undefined;

      return {
        historyCount: history.length,
        estimatedSize: `${sizeInKB} KB`,
        lastModified,
      };
    } catch (error) {
      console.error('Failed to calculate storage stats:', error);
      return { historyCount: 0, estimatedSize: '0 KB' };
    }
  }
}

export const storage = new StorageManager();

// Utility functions for file handling
export const createThumbnail = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          // Set thumbnail dimensions
          const maxSize = 200;
          let { width, height } = img;
          
          if (width > height) {
            if (width > maxSize) {
              height = (height * maxSize) / width;
              width = maxSize;
            }
          } else {
            if (height > maxSize) {
              width = (width * maxSize) / height;
              height = maxSize;
            }
          }
          
          canvas.width = width;
          canvas.height = height;
          
          ctx?.drawImage(img, 0, 0, width, height);
          resolve(canvas.toDataURL('image/jpeg', 0.7));
        };
        img.onerror = reject;
        img.src = e.target?.result as string;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        video.currentTime = Math.min(video.duration * 0.1, 5); // 10% or 5 seconds
      };
      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = Math.min(video.videoWidth, 200);
        canvas.height = Math.min(video.videoHeight, 200);
        
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.7));
      };
      video.onerror = reject;
      video.src = URL.createObjectURL(file);
    } else {
      // For audio and other files, return a placeholder
      resolve('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSIyMDAiIGZpbGw9IiNmM2Y0ZjYiLz48cGF0aCBkPSJNMTAwIDUwSDEyMFY3MEgxMDBWNTBaTTEwMCA4MEgxMjBWMTAwSDEwMFY4MFpNMTAwIDExMEgxMjBWMTMwSDEwMFYxMTBaTTEwMCAxNDBIMTIwVjE2MEgxMDBWMTQwWiIgZmlsbD0iIzZiNzI4MCIvPjwvc3ZnPg==');
    }
  });
};

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const getFileIcon = (fileType: string): string => {
  if (fileType.startsWith('image/')) return '🖼️';
  if (fileType.startsWith('video/')) return '🎥';
  if (fileType.startsWith('audio/')) return '🎵';
  return '📄';
};