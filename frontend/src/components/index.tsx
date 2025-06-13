import { useState } from 'react';
import { Container, Row, Col, Alert, Card } from 'react-bootstrap';
import URLForm from '@/components/URLForm';
import PreviewCard from '@/components/PreviewCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import GenerateAd from '@/components/GenerateAd';
import ErrorAlert from '@/components/ErrorAlert';

type Product = {
  title: string;
  price: string;
  description?: string;
  features: string[];
  images: string[];
  [key: string]: any;
};

export default function Home() {
  const [product, setProduct] = useState<Product | null>(null);
  const [jobId, setJobId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const clearError = () => setError('');

  return (
    <Container fluid className="min-vh-100 bg-light py-4">
      <Row className="justify-content-center">
        <Col lg={10} xl={8}>
          <Card className="shadow-sm border-0 mb-4">
            <Card.Header className="bg-primary text-white text-center py-3">
              <h1 className="h2 mb-0">🎬 AI Video Ad Generator Dashboard</h1>
            </Card.Header>
            <Card.Body className="p-4">
              <URLForm 
                setProduct={setProduct} 
                setLoading={setLoading} 
                setError={setError}
              />
              
              {error && (
                <ErrorAlert message={error} onClose={clearError} />
              )}
              
              {loading && <LoadingSpinner />}
              
              {product && !loading && (
                <Row className="mt-4">
                  <Col lg={6}>
                    <PreviewCard product={product} />
                  </Col>
                  <Col lg={6}>
                    <GenerateAd 
                      product={product} 
                      setJobId={setJobId} 
                      setError={setError}
                    />
                  </Col>
                </Row>
              )}
              
              {jobId && (
                <Alert variant="success" className="mt-4 text-center">
                  <Alert.Heading className="h5">✅ Ad Generated Successfully!</Alert.Heading>
                  <p className="mb-2">Job ID: <code>{jobId}</code></p>
                  <Alert.Link href={`/preview?job_id=${jobId}`} className="btn btn-outline-success">
                    View Preview & Download
                  </Alert.Link>
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}
