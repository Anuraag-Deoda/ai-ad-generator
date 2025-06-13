import { Dispatch, SetStateAction, useState } from 'react';
import { Card, Button, Spinner, Form, Row, Col, ButtonGroup } from 'react-bootstrap';
import { motion } from 'framer-motion';

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

  const styleOptions = [
    { value: 'energetic', label: 'Energetic', icon: '⚡', description: 'High-energy, exciting tone' },
    { value: 'professional', label: 'Professional', icon: '💼', description: 'Business-focused approach' },
    { value: 'casual', label: 'Casual', icon: '😊', description: 'Friendly, approachable style' },
    { value: 'luxury', label: 'Luxury', icon: '✨', description: 'Premium, sophisticated feel' }
  ];

  const durationOptions = [
    { value: 15, label: '15s', description: 'Quick & Punchy', recommended: false },
    { value: 30, label: '30s', description: 'Perfect Balance', recommended: true },
    { value: 60, label: '60s', description: 'Detailed Story', recommended: false }
  ];

  return (
    <>
      <style jsx>{`
        .generate-card {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          transition: all 0.3s ease;
          height: 100%;
        }
        
        .generate-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        }
        
        .card-header-custom {
          background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
          color: white;
          padding: 16px 20px;
          border: none;
        }
        
        .header-title {
          font-size: 1.1rem;
          font-weight: 600;
          margin: 0;
          display: flex;
          align-items: center;
        }
        
        .header-icon {
          margin-right: 8px;
          font-size: 1.3rem;
        }
        
        .section-title {
          font-size: 1rem;
          font-weight: 600;
          color: #495057;
          margin-bottom: 12px;
          display: flex;
          align-items: center;
        }
        
        .section-icon {
          margin-right: 8px;
          color: #0d6efd;
        }
        
        .style-option {
          border: 2px solid #e9ecef;
          border-radius: 8px;
          padding: 12px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s ease;
          background: #fff;
          height: 100%;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }
        
        .style-option:hover {
          border-color: #0d6efd;
          transform: translateY(-2px);
          box-shadow: 0 4px 15px rgba(13, 110, 253, 0.15);
        }
        
        .style-option.selected {
          border-color: #0d6efd;
          background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
          box-shadow: 0 4px 15px rgba(13, 110, 253, 0.2);
        }
        
        .style-icon {
          font-size: 1.5rem;
          margin-bottom: 8px;
        }
        
        .style-label {
          font-weight: 600;
          color: #212529;
          margin-bottom: 4px;
        }
        
        .style-description {
          font-size: 0.8rem;
          color: #6c757d;
        }
        
        .duration-option {
          border: 2px solid #e9ecef;
          border-radius: 8px;
          padding: 12px 16px;
          cursor: pointer;
          transition: all 0.3s ease;
          background: #fff;
          margin-bottom: 8px;
          position: relative;
        }
        
        .duration-option:hover {
          border-color: #0d6efd;
          transform: translateX(4px);
        }
        
        .duration-option.selected {
          border-color: #0d6efd;
          background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        }
        
        .duration-label {
          font-weight: 600;
          color: #212529;
          font-size: 1rem;
        }
        
        .duration-description {
          color: #6c757d;
          font-size: 0.85rem;
        }
        
        .recommended-badge {
          position: absolute;
          top: -6px;
          right: 8px;
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          color: white;
          font-size: 0.7rem;
          padding: 2px 8px;
          border-radius: 10px;
          font-weight: 600;
        }
        
        .generate-button {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          border: none;
          border-radius: 8px;
          padding: 14px 24px;
          font-weight: 600;
          font-size: 1.1rem;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
          width: 100%;
        }
        
        .generate-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
          background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
        }
        
        .generate-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }
        
        .generating-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(255, 255, 255, 0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 12px;
          z-index: 10;
        }
        
        .generating-content {
          text-align: center;
          color: #495057;
        }
        
        .ai-info {
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border: 1px solid #e9ecef;
          border-radius: 8px;
          padding: 12px;
          margin-top: 16px;
          text-align: center;
        }
        
        .ai-info-text {
          color: #6c757d;
          font-size: 0.85rem;
          margin: 0;
        }
      `}</style>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        style={{ position: 'relative' }}
      >
        <Card className="generate-card">
          <div className="card-header-custom">
            <h5 className="header-title">
              <i className="bi bi-magic header-icon"></i>
              Generate Video Ad
            </h5>
          </div>
          
          <Card.Body className="p-3">
            <Form>
              {/* Style Selection */}
              <div className="mb-4">
                <h6 className="section-title">
                  <i className="bi bi-palette section-icon"></i>
                  Ad Style
                </h6>
                <Row>
                  {styleOptions.map((option) => (
                    <Col xs={6} key={option.value} className="mb-3">
                      <div
                        className={`style-option ${style === option.value ? 'selected' : ''}`}
                        onClick={() => setStyle(option.value)}
                      >
                        <div className="style-icon">{option.icon}</div>
                        <div className="style-label">{option.label}</div>
                        <div className="style-description">{option.description}</div>
                      </div>
                    </Col>
                  ))}
                </Row>
              </div>
              
              {/* Duration Selection */}
              <div className="mb-4">
                <h6 className="section-title">
                  <i className="bi bi-clock section-icon"></i>
                  Video Duration
                </h6>
                {durationOptions.map((option) => (
                  <div
                    key={option.value}
                    className={`duration-option ${duration === option.value ? 'selected' : ''}`}
                    onClick={() => setDuration(option.value)}
                  >
                    {option.recommended && (
                      <div className="recommended-badge">Recommended</div>
                    )}
                    <div className="d-flex justify-content-between align-items-center">
                      <div>
                        <div className="duration-label">{option.label}</div>
                        <div className="duration-description">{option.description}</div>
                      </div>
                      <i className={`bi bi-${duration === option.value ? 'check-circle-fill' : 'circle'}`}></i>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Generate Button */}
              <Button
                className="generate-button"
                onClick={handleGenerate}
                disabled={generating}
              >
                {generating ? (
                  <>
                    <Spinner size="sm" className="me-2" />
                    Creating Magic...
                  </>
                ) : (
                  <>
                    <i className="bi bi-play-circle me-2"></i>
                    Generate Video Ad
                  </>
                )}
              </Button>
              
              {!generating && (
                <div className="ai-info">
                  <p className="ai-info-text">
                    <i className="bi bi-cpu me-1"></i>
                    AI will create a custom script and video for your product
                  </p>
                </div>
              )}
            </Form>
          </Card.Body>
          
          {generating && (
            <div className="generating-overlay">
              <div className="generating-content">
                <Spinner animation="border" variant="primary" className="mb-3" />
                <h5>Generating Your Video Ad</h5>
                <p className="text-muted">This may take 1-3 minutes...</p>
              </div>
            </div>
          )}
        </Card>
      </motion.div>
    </>
  );
}

