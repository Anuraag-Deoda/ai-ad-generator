// frontend/src/pages/preview.tsx
import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import { Container, Row, Col, Card, Spinner, Alert, Badge } from 'react-bootstrap';
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
      <Container className="py-5">
        <Row className="justify-content-center">
          <Col md={6} className="text-center">
            <Spinner animation="border" variant="primary" className="mb-3" />
            <h4>Loading preview...</h4>
          </Col>
        </Row>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="py-5">
        <Row className="justify-content-center">
          <Col lg={8}>
            <ErrorAlert message={error} onClose={() => setError('')} />
          </Col>
        </Row>
      </Container>
    );
  }

  if (!jobData) {
    return (
      <Container className="py-5">
        <Row className="justify-content-center">
          <Col md={6} className="text-center">
            <Alert variant="warning">
              <h5>No preview data available</h5>
              <p>The requested job could not be found.</p>
            </Alert>
          </Col>
        </Row>
      </Container>
    );
  }

  return (
    <Container fluid className="py-4 bg-light min-vh-100">
      <Row className="justify-content-center">
        <Col lg={10} xl={8}>
          <Card className="shadow-sm border-0">
            <Card.Header className="bg-primary text-white">
              <Row className="align-items-center">
                <Col>
                  <h1 className="h3 mb-0">🎬 {jobData.title}</h1>
                </Col>
                <Col xs="auto">
                  <Badge bg="light" text="dark">Job ID: {job_id}</Badge>
                </Col>
              </Row>
            </Card.Header>
            
            <Card.Body className="p-4">
              <Row>
                <Col lg={8}>
                  <Card className="mb-4">
                    <Card.Header>
                      <h5 className="mb-0">📹 Generated Video</h5>
                      {videoGenerating && (
                        <small className="text-muted">
                          <Spinner size="sm" className="me-1" />
                          Video is being generated...
                        </small>
                      )}
                    </Card.Header>
                    <Card.Body>
                      <VideoPlayer jobId={job_id as string} />
                    </Card.Body>
                  </Card>
                </Col>
                
                <Col lg={4}>
                  <Card>
                    <Card.Header>
                      <h5 className="mb-0">📝 Ad Script</h5>
                    </Card.Header>
                    <Card.Body>
                      <div className="mb-3">
                        <h6 className="text-primary">Hook</h6>
                        <p className="small">{jobData.script.hook}</p>
                      </div>
                      
                      <div className="mb-3">
                        <h6 className="text-primary">Pitch</h6>
                        <p className="small">{jobData.script.pitch}</p>
                      </div>
                      
                      <div className="mb-3">
                        <h6 className="text-primary">Features</h6>
                        <p className="small">{jobData.script.features}</p>
                      </div>
                      
                      <div>
                        <h6 className="text-primary">Call to Action</h6>
                        <p className="small">{jobData.script.cta}</p>
                      </div>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}
