// frontend/src/components/GenerateAd.tsx
import { Dispatch, SetStateAction, useState } from 'react';
import { Card, Button, Spinner, Form, Row, Col } from 'react-bootstrap';

type Product = {
  title: string;
  price: string;
  features: string[];
  description?: string;
  images: string[];
};

type Props = {
  product: Product;
  setJobId: Dispatch<SetStateAction<string>>;
  setError: Dispatch<SetStateAction<string>>;
};

export default function GenerateAd({ product, setJobId, setError }: Props) {
  const [generating, setGenerating] = useState(false);
  const [style, setStyle] = useState('energetic');
  const [duration, setDuration] = useState(30);

  const handleGenerate = async () => {
    setGenerating(true);
    setError('');
    
    try {
      const jobId = `job_${Date.now()}`;
      const res = await fetch('http://localhost:5000/api/generate-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          product,
          style,
          duration,
          include_metadata: true,
          include_variations: false
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      if (data.success && data.job_id) {
        setJobId(data.job_id);
      } else {
        setError(data.error || 'Failed to generate ad. Please try again.');
      }
    } catch (err) {
      console.error('Generation failed:', err);
      setError('Unable to generate ad. Please check your connection and try again.');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <Card className="h-100 shadow-sm">
      <Card.Header className="bg-light">
        <h5 className="mb-0">🎯 Generate Video Ad</h5>
      </Card.Header>
      <Card.Body>
        <Form>
          <Row>
            <Col md={6}>
              <Form.Group className="mb-3">
                <Form.Label>Ad Style</Form.Label>
                <Form.Select 
                  value={style} 
                  onChange={(e) => setStyle(e.target.value)}
                  disabled={generating}
                >
                  <option value="energetic">Energetic</option>
                  <option value="professional">Professional</option>
                  <option value="casual">Casual</option>
                  <option value="luxury">Luxury</option>
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group className="mb-3">
                <Form.Label>Duration (seconds)</Form.Label>
                <Form.Select 
                  value={duration} 
                  onChange={(e) => setDuration(Number(e.target.value))}
                  disabled={generating}
                >
                  <option value={15}>15 seconds</option>
                  <option value={30}>30 seconds</option>
                  <option value={60}>60 seconds</option>
                </Form.Select>
              </Form.Group>
            </Col>
          </Row>
          
          <div className="d-grid">
            <Button
              variant="success"
              size="lg"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? (
                <>
                  <Spinner size="sm" className="me-2" />
                  Generating Ad...
                </>
              ) : (
                '🚀 Generate Video Ad'
              )}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
}
