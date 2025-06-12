import { useState } from 'react';
import { Button, Alert, Spinner } from 'react-bootstrap';

type Props = {
  jobId: string;
};

export default function VideoPlayer({ jobId }: Props) {
  const [videoError, setVideoError] = useState(false);
  const [videoLoading, setVideoLoading] = useState(true);

  const handleVideoLoad = () => setVideoLoading(false);
  const handleVideoError = () => {
    setVideoLoading(false);
    setVideoError(true);
  };

  const videoUrl = `http://localhost:5000/static/videos/${jobId}.mp4`;

  if (videoError) {
    return (
      <Alert variant="warning" className="text-center">
        <h6>Video not ready yet</h6>
        <p className="mb-2">The video is still being processed. Please wait a moment and refresh the page.</p>
        <Button variant="outline-warning" size="sm" onClick={() => window.location.reload()}>
          Refresh Page
        </Button>
      </Alert>
    );
  }

  return (
    <div className="position-relative">
      {videoLoading && (
        <div className="text-center py-5">
          <Spinner animation="border" variant="primary" className="mb-2" />
          <p className="text-muted">Loading video...</p>
        </div>
      )}
      
      <video
        controls
        className={`w-100 rounded ${videoLoading ? 'd-none' : ''}`}
        src={videoUrl}
        onLoadedData={handleVideoLoad}
        onError={handleVideoError}
        style={{ maxHeight: '400px' }}
      />
      
      {!videoLoading && !videoError && (
        <div className="text-center mt-3">
          <Button
            variant="outline-primary"
            href={videoUrl}
            download={`ad-video-${jobId}.mp4`}
            target="_blank"
          >
            <i className="bi bi-download me-2"></i>
            Download Video
          </Button>
        </div>
      )}
    </div>
  );
}
