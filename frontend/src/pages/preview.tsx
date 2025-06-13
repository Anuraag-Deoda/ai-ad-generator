import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import { Container, Row, Col, Card, Spinner, Alert, Badge, Button } from 'react-bootstrap';
import { motion } from 'framer-motion';
import DashboardLayout from '@/components/DashboardLayout';
import VideoPlayer from '@/components/VideoPlayer';
import ErrorAlert from '@/components/ErrorAlert';

type Script = {
  hook: string;
  pitch: string;
  features: string;
  cta: string;
};

type JobData = {
  title: string;
  script: Script;
  video_path?: string;
};

export default function PreviewPage() {
  const router = useRouter();
  const { job_id } = router.query;

  const [jobData, setJobData] = useState<JobData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [videoGenerating, setVideoGenerating] = useState(false);

  useEffect(() => {
    if (!job_id) return;

    setVideoGenerating(true);
    fetch('http://localhost:5000/api/generate-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          console.log('🎬 Video rendered:', data.video_path);
        } else {
          console.error('Video generation error:', data.error);
          setError('Failed to generate video: ' + (data.error || 'Unknown error'));
        }
      })
      .catch(err => {
        console.error('Video generation request failed:', err);
        setError('Failed to connect to video generation service');
      })
      .finally(() => {
        setVideoGenerating(false);
      });
  }, [job_id]);

  useEffect(() => {
    if (!job_id) return;

    fetch(`http://localhost:5000/api/job-preview/${job_id}`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        if (data.success) {
          setJobData(data.data);
        } else {
          setError('Failed to load preview data');
        }
      })
      .catch(err => {
        console.error('Failed to load preview:', err);
        setError('Failed to load preview data');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [job_id]);

  if (loading) {
    return (
      <>
        <style jsx global>{`
          .loading-container {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
          }
          
          .loading-card {
            background: #fff;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 40px;
            text-align: center;
            max-width: 400px;
          }
          
          .loading-icon {
            font-size: 3rem;
            margin-bottom: 20px;
            animation: bounce 2s infinite;
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
          }
        `}</style>
        
        <DashboardLayout>
          <div className="loading-container">
            <Card className="loading-card">
              <div className="loading-icon">🎬</div>
              <Spinner animation="border" variant="primary" className="mb-3" style={{ width: '3rem', height: '3rem' }} />
              <h4 className="text-primary mb-2">Loading Preview</h4>
              <p className="text-muted">Preparing your video ad...</p>
            </Card>
          </div>
        </DashboardLayout>
      </>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <Container>
          <Row className="justify-content-center">
            <Col lg={8}>
              <ErrorAlert message={error} onClose={() => setError('')} />
            </Col>
          </Row>
        </Container>
      </DashboardLayout>
    );
  }

  if (!jobData) {
    return (
      <DashboardLayout>
        <Container>
          <Row className="justify-content-center">
            <Col md={6} className="text-center">
              <Alert variant="warning" className="border-0 shadow-sm" style={{ borderRadius: '12px' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: '15px' }}>🔍</div>
                <h5>No Preview Available</h5>
                <p>The requested job could not be found.</p>
                <Button variant="primary" href="/">
                  <i className="bi bi-arrow-left me-2"></i>
                  Back to Dashboard
                </Button>
              </Alert>
            </Col>
          </Row>
        </Container>
      </DashboardLayout>
    );
  }

  return (
    <>
      <style jsx global>{`
        .preview-header {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          margin-bottom: 24px;
          overflow: hidden;
        }
        
        .header-gradient {
          background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
          color: white;
          padding: 20px 24px;
        }
        
        .preview-title {
          font-size: 1.5rem;
          font-weight: 700;
          margin: 0;
          display: flex;
          align-items: center;
        }
        
        .title-icon {
          margin-right: 12px;
          font-size: 1.8rem;
        }
        
        .job-badge {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
          padding: 8px 16px;
          border-radius: 20px;
          font-size: 0.85rem;
          font-weight: 600;
        }
        
        .video-section {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          margin-bottom: 24px;
        }
        
        .section-header {
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-bottom: 1px solid #e9ecef;
          padding: 16px 20px;
          display: flex;
          justify-content: between;
          align-items: center;
        }
        
        .section-title {
          font-size: 1.1rem;
          font-weight: 600;
          color: #212529;
          margin: 0;
          display: flex;
          align-items: center;
        }
        
        .section-icon {
          margin-right: 8px;
          color: #0d6efd;
        }
        
        .status-badge {
          background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
          color: #212529;
          font-size: 0.75rem;
          padding: 4px 8px;
          border-radius: 12px;
          font-weight: 600;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
        
        .script-section {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          overflow: hidden;
        }
        
        .script-item {
          padding: 16px 20px;
          border-bottom: 1px solid #f8f9fa;
          transition: all 0.3s ease;
        }
        
        .script-item:last-child {
          border-bottom: none;
        }
        
        .script-item:hover {
          background: #f8f9fa;
        }
        
        .script-label {
          font-size: 0.9rem;
          font-weight: 600;
          color: #0d6efd;
          margin-bottom: 8px;
          display: flex;
          align-items: center;
        }
        
        .script-label-icon {
          margin-right: 6px;
        }
        
        .script-text {
          color: #495057;
          line-height: 1.6;
          font-size: 0.95rem;
          margin: 0;
        }
        
        .breadcrumb-nav {
          background: transparent;
          padding: 0;
          margin-bottom: 20px;
        }
        
        .breadcrumb-item {
          color: #6c757d;
        }
        
        .breadcrumb-item.active {
          color: #0d6efd;
          font-weight: 600;
        }
      `}</style>

      <DashboardLayout>
        {/* Breadcrumb Navigation */}
        <nav className="breadcrumb-nav">
          <ol className="breadcrumb">
            <li className="breadcrumb-item">
              <a href="/" className="text-decoration-none">
                <i className="bi bi-house me-1"></i>
                Dashboard
              </a>
            </li>
            <li className="breadcrumb-item active">Video Preview</li>
          </ol>
        </nav>

        {/* Preview Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="preview-header">
            <div className="header-gradient">
              <Row className="align-items-center">
                <Col>
                  <h1 className="preview-title">
                    <i className="bi bi-play-circle title-icon"></i>
                    {jobData.title}
                  </h1>
                  <p className="mb-0 opacity-90">
                    AI-generated video advertisement ready for download
                  </p>
                </Col>
                <Col xs="auto">
                  <Badge className="job-badge">
                    Job ID: {job_id}
                  </Badge>
                </Col>
              </Row>
            </div>
          </Card>
        </motion.div>

        <Row>
          {/* Video Section */}
          <Col lg={8} className="mb-4">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card className="video-section">
                <div className="section-header">
                  <h5 className="section-title">
                    <i className="bi bi-camera-video section-icon"></i>
                    Generated Video
                  </h5>
                  {videoGenerating && (
                    <Badge className="status-badge">
                      <Spinner size="sm" className="me-1" />
                      Processing...
                    </Badge>
                  )}
                </div>
                <Card.Body className="p-0">
                  <VideoPlayer jobId={job_id as string} />
                </Card.Body>
              </Card>
            </motion.div>
          </Col>
          
          {/* Script Section */}
          <Col lg={4}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <Card className="script-section">
                <div className="section-header">
                  <h5 className="section-title">
                    <i className="bi bi-file-text section-icon"></i>
                    Ad Script
                  </h5>
                </div>
                
                <div className="script-item">
                  <h6 className="script-label">
                    <i className="bi bi-bullseye script-label-icon"></i>
                    Hook
                  </h6>
                  <p className="script-text">{jobData.script.hook}</p>
                </div>
                
                <div className="script-item">
                  <h6 className="script-label">
                    <i className="bi bi-chat-dots script-label-icon"></i>
                    Pitch
                  </h6>
                  <p className="script-text">{jobData.script.pitch}</p>
                </div>
                
                <div className="script-item">
                  <h6 className="script-label">
                    <i className="bi bi-star script-label-icon"></i>
                    Features
                  </h6>
                  <p className="script-text">{jobData.script.features}</p>
                </div>
                
                <div className="script-item">
                  <h6 className="script-label">
                    <i className="bi bi-megaphone script-label-icon"></i>
                    Call to Action
                  </h6>
                  <p className="script-text">{jobData.script.cta}</p>
                </div>
              </Card>
            </motion.div>
          </Col>
        </Row>
      </DashboardLayout>
    </>
  );
}

