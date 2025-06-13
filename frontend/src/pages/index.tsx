import { useState } from 'react';
import { Container, Row, Col, Card, Alert, Badge } from 'react-bootstrap';
import { motion } from 'framer-motion';
import DashboardLayout from '@/components/DashboardLayout';
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

export default function Dashboard() {
  const [product, setProduct] = useState<Product | null>(null);
  const [jobId, setJobId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const clearError = () => setError('');

  return (
    <>
      <style jsx global>{`
        .dashboard-header {
          background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
          margin-bottom: 24px;
        }
        
        .stats-card {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          transition: all 0.3s ease;
          height: 100%;
        }
        
        .stats-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        }
        
        .stats-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.5rem;
          color: white;
          margin-bottom: 16px;
        }
        
        .stats-icon.primary {
          background: linear-gradient(135deg, #0d6efd 0%, #0056b3 100%);
        }
        
        .stats-icon.success {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        }
        
        .stats-icon.warning {
          background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        }
        
        .stats-icon.info {
          background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        }
        
        .stats-number {
          font-size: 2rem;
          font-weight: 700;
          color: #212529;
          margin-bottom: 4px;
        }
        
        .stats-label {
          color: #6c757d;
          font-size: 0.9rem;
          font-weight: 500;
        }
        
        .workflow-card {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          margin-bottom: 24px;
          overflow: hidden;
        }
        
        .workflow-header {
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-bottom: 1px solid #e9ecef;
          padding: 20px 24px;
        }
        
        .workflow-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #212529;
          margin: 0;
          display: flex;
          align-items: center;
        }
        
        .workflow-icon {
          margin-right: 12px;
          font-size: 1.5rem;
        }
        
        .success-alert {
          background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
          border: 1px solid #c3e6cb;
          border-radius: 12px;
          color: #155724;
        }
        
        .success-link {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          color: white;
          text-decoration: none;
          padding: 12px 24px;
          border-radius: 8px;
          font-weight: 600;
          transition: all 0.3s ease;
          display: inline-block;
        }
        
        .success-link:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(40, 167, 69, 0.3);
          color: white;
          text-decoration: none;
        }
        
        .page-title {
          font-size: 2rem;
          font-weight: 700;
          color: #212529;
          margin-bottom: 8px;
        }
        
        .page-subtitle {
          color: #6c757d;
          font-size: 1.1rem;
          margin-bottom: 0;
        }
        
        .content-section {
          animation: slideInUp 0.6s ease-out;
        }
        
        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      <DashboardLayout>
        {/* Dashboard Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="dashboard-header">
            <Card.Body className="p-4">
              <Row className="align-items-center">
                <Col>
                  <h1 className="page-title">AI Video Ad Generator</h1>
                  <p className="page-subtitle">
                    Transform any product URL into stunning video advertisements
                  </p>
                </Col>
                <Col xs="auto">
                  <Badge bg="primary" className="px-3 py-2">
                    <i className="bi bi-lightning-charge me-1"></i>
                    AI Powered
                  </Badge>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </motion.div>


        {/* Main Workflow */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card className="workflow-card">
            <div className="workflow-header">
              <h3 className="workflow-title">
                <i className="bi bi-magic workflow-icon"></i>
                Create New Video Ad
              </h3>
            </div>
            <Card.Body className="p-4">
              <div className="content-section">
                <URLForm 
                  setProduct={setProduct} 
                  setLoading={setLoading} 
                  setError={setError}
                />
                
                {error && (
                  <div style={{ animationDelay: '0.2s' }} className="content-section">
                    <ErrorAlert message={error} onClose={clearError} />
                  </div>
                )}
                
                {loading && (
                  <div style={{ animationDelay: '0.4s' }} className="content-section">
                    <LoadingSpinner />
                  </div>
                )}
                
                {product && !loading && (
                  <Row className="mt-4" style={{ animationDelay: '0.6s' }}>
                    <Col lg={6} className="mb-4">
                      <div className="content-section">
                        <PreviewCard product={product} />
                      </div>
                    </Col>
                    <Col lg={6} className="mb-4">
                      <div className="content-section">
                        <GenerateAd 
                          product={product} 
                          setJobId={setJobId} 
                          setError={setError}
                        />
                      </div>
                    </Col>
                  </Row>
                )}
                
                {jobId && (
                  <div style={{ animationDelay: '0.8s' }} className="content-section mt-4">
                    <Alert className="success-alert text-center py-4">
                      <div className="mb-3" style={{ fontSize: '3rem' }}>🎉</div>
                      <Alert.Heading className="h4 mb-3">
                        Video Ad Generated Successfully!
                      </Alert.Heading>
                      <p className="mb-3">
                        Job ID: <code className="bg-white px-2 py-1 rounded">{jobId}</code>
                      </p>
                      <a 
                        href={`/preview?job_id=${jobId}`} 
                        className="success-link"
                      >
                        <i className="bi bi-play-circle me-2"></i>
                        View Preview & Download
                      </a>
                    </Alert>
                  </div>
                )}
              </div>
            </Card.Body>
          </Card>
        </motion.div>
      </DashboardLayout>
    </>
  );
}

