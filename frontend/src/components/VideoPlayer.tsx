import { useState } from 'react';
import { Button, Alert, Spinner, Card, Row, Col, Badge } from 'react-bootstrap';
import { motion } from 'framer-motion';

type Props = {
  jobId: string;
};

export default function VideoPlayer({ jobId }: Props) {
  const [videoError, setVideoError] = useState(false);
  const [videoLoading, setVideoLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

  const handleVideoLoad = () => setVideoLoading(false);
  const handleVideoError = () => {
    setVideoLoading(false);
    setVideoError(true);
  };

  const handleRetry = () => {
    setVideoError(false);
    setVideoLoading(true);
    setRetryCount(prev => prev + 1);
  };

  const videoUrl = `http://localhost:5000/static/videos/${jobId}.mp4?retry=${retryCount}`;

  if (videoError) {
    return (
      <>
        <style jsx>{`
          .video-error-card {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffeaa7;
            border-radius: 12px;
            color: #856404;
            text-align: center;
            padding: 32px 24px;
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.15);
          }
          
          .error-icon {
            font-size: 3rem;
            margin-bottom: 16px;
            animation: bounce 2s infinite;
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-8px); }
            60% { transform: translateY(-4px); }
          }
          
          .error-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #856404;
            margin-bottom: 12px;
          }
          
          .error-description {
            color: #856404;
            margin-bottom: 20px;
            line-height: 1.5;
          }
          
          .refresh-button {
            background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);
            border: none;
            color: #212529;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
          }
          
          .refresh-button:hover {
            background: linear-gradient(135deg, #ffb300 0%, #ff8f00 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 193, 7, 0.4);
            color: #212529;
          }
        `}</style>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="video-error-card">
            <div className="error-icon">🎬</div>
            <h5 className="error-title">Video Processing in Progress</h5>
            <p className="error-description">
              Your video is still being crafted by our AI. This usually takes 1-3 minutes.
              Please wait a moment and refresh to check the status.
            </p>
            <Button
              className="refresh-button"
              onClick={handleRetry}
            >
              <i className="bi bi-arrow-clockwise me-2"></i>
              Check Again
            </Button>
          </Card>
        </motion.div>
      </>
    );
  }

  return (
    <>
      <style jsx>{`
        .video-container {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
          transition: all 0.3s ease;
        }
        
        .video-container:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        }
        
        .video-wrapper {
          position: relative;
          background: #000;
          border-radius: 12px 12px 0 0;
          overflow: hidden;
        }
        
        .video-loading {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          color: white;
          z-index: 2;
          min-height: 300px;
        }
        
        .loading-spinner {
          width: 48px;
          height: 48px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-top: 3px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 16px;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .loading-text {
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 4px;
        }
        
        .loading-subtext {
          font-size: 0.9rem;
          opacity: 0.8;
        }
        
        .video-element {
          width: 100%;
          max-height: 400px;
          border-radius: 12px 12px 0 0;
          transition: all 0.3s ease;
        }
        
        .video-controls {
          padding: 20px;
          background: #fff;
        }
        
        .video-info {
          margin-bottom: 16px;
        }
        
        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }
        
        .info-item {
          text-align: center;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 8px;
          border: 1px solid #e9ecef;
        }
        
        .info-icon {
          color: #0d6efd;
          font-size: 1.1rem;
          margin-bottom: 4px;
        }
        
        .info-label {
          font-size: 0.8rem;
          color: #6c757d;
          font-weight: 500;
        }
        
        .download-section {
          text-align: center;
        }
        
        .download-button {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          border: none;
          border-radius: 8px;
          padding: 12px 24px;
          font-weight: 600;
          color: white;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        
        .download-button:hover {
          background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
          color: white;
        }
        
        .download-info {
          margin-top: 12px;
          color: #6c757d;
          font-size: 0.85rem;
        }
      `}</style>
      
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="video-container">
          <div className="video-wrapper">
            {videoLoading && (
              <div className="video-loading">
                <div className="loading-spinner"></div>
                <div className="loading-text">Loading your video...</div>
                <div className="loading-subtext">Almost ready!</div>
              </div>
            )}
            
            <video
              controls
              className={`video-element ${videoLoading ? 'd-none' : ''}`}
              src={videoUrl}
              onLoadedData={handleVideoLoad}
              onError={handleVideoError}
              poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='400'%3E%3Crect width='100%25' height='100%25' fill='%23000'/%3E%3Ctext x='50%25' y='50%25' font-size='24' fill='white' text-anchor='middle' dy='.3em'%3E🎬 Your Video Ad%3C/text%3E%3C/svg%3E"
            />
          </div>
          
          {!videoLoading && !videoError && (
            <div className="video-controls">
              <div className="video-info">
                <div className="info-grid">
                  <div className="info-item">
                    <div className="info-icon">
                      <i className="bi bi-camera-video"></i>
                    </div>
                    <div className="info-label">HD Quality</div>
                  </div>
                  <div className="info-item">
                    <div className="info-icon">
                      <i className="bi bi-file-earmark-play"></i>
                    </div>
                    <div className="info-label">MP4 Format</div>
                  </div>
                  <div className="info-item">
                    <div className="info-icon">
                      <i className="bi bi-check-circle"></i>
                    </div>
                    <div className="info-label">Ready to Share</div>
                  </div>
                  <div className="info-item">
                    <div className="info-icon">
                      <i className="bi bi-phone"></i>
                    </div>
                    <div className="info-label">Mobile Ready</div>
                  </div>
                </div>
              </div>
              
              <div className="download-section">
                <Button
                  className="download-button"
                  href={videoUrl}
                  download={`ad-video-${jobId}.mp4`}
                  target="_blank"
                >
                  <i className="bi bi-download me-2"></i>
                  Download Video
                </Button>
                <div className="download-info">
                  <i className="bi bi-info-circle me-1"></i>
                  Perfect for social media, websites, and marketing campaigns
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </>
  );
}

