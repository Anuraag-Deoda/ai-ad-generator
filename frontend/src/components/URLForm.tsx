import { Dispatch, SetStateAction, useState } from 'react';
import { Form, Button, InputGroup, Spinner, Card, Row, Col } from 'react-bootstrap';
import { motion } from 'framer-motion';

type Product = {
  title: string;
  price: string;
  description?: string;
  features: string[];
  images: string[];
  [key: string]: any;
};

type Props = {
  setProduct: Dispatch<SetStateAction<Product | null>>;
  setLoading: Dispatch<SetStateAction<boolean>>;
  setError: Dispatch<SetStateAction<string>>;
};

export default function URLForm({ setProduct, setLoading, setError }: Props) {
  const [url, setUrl] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const validateUrl = (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    
    if (!url.trim()) {
      setError('Please enter a product URL');
      return;
    }
    
    if (!validateUrl(url)) {
      setError('Please enter a valid URL');
      return;
    }

    setIsSubmitting(true);
    setLoading(true);
    
    try {
      const res = await fetch('http://localhost:5000/api/analyze-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      if (data.success) {
        setProduct(data);
        setUrl('');
      } else {
        setError(data.error || 'Failed to analyze product. Please check the URL and try again.');
      }
    } catch (err) {
      console.error('Request failed', err);
      setError('Unable to connect to server. Please check your connection and try again.');
    } finally {
      setLoading(false);
      setIsSubmitting(false);
    }
  };

  const exampleUrls = [
    { label: 'Amazon Product', url: 'https://www.amazon.com/dp/B08N5WRWNW' },
    { label: 'Shopify Store', url: 'https://shopify-store.com/products/item' },
    { label: 'Etsy Item', url: 'https://www.etsy.com/listing/123456789' },
  ];

  return (
    <>
      <style jsx>{`
        .url-form-container {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          padding: 24px;
          margin-bottom: 24px;
        }
        
        .form-header {
          margin-bottom: 20px;
          text-align: center;
        }
        
        .form-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #212529;
          margin-bottom: 8px;
        }
        
        .form-subtitle {
          color: #6c757d;
          font-size: 0.95rem;
        }
        
        .url-input {
          border: 2px solid #e9ecef;
          border-radius: 8px;
          padding: 12px 16px;
          font-size: 1rem;
          transition: all 0.3s ease;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .url-input:focus {
          border-color: #0d6efd;
          box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.15);
        }
        
        .analyze-button {
          background: linear-gradient(135deg, #0d6efd 0%, #0056b3 100%);
          border: none;
          border-radius: 8px;
          padding: 12px 24px;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(13, 110, 253, 0.3);
        }
        
        .analyze-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(13, 110, 253, 0.4);
          background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
        }
        
        .analyze-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }
        
        .examples-section {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #e9ecef;
        }
        
        .examples-title {
          font-size: 0.9rem;
          font-weight: 600;
          color: #495057;
          margin-bottom: 12px;
          display: flex;
          align-items: center;
        }
        
        .example-chip {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 20px;
          padding: 8px 16px;
          font-size: 0.85rem;
          color: #495057;
          text-decoration: none;
          transition: all 0.3s ease;
          display: inline-block;
          margin: 4px;
          cursor: pointer;
        }
        
        .example-chip:hover {
          background: #e9ecef;
          border-color: #0d6efd;
          color: #0d6efd;
          transform: translateY(-1px);
          text-decoration: none;
        }
        
        .input-icon {
          color: #6c757d;
          font-size: 1.1rem;
        }
        
        .loading-state {
          pointer-events: none;
          opacity: 0.7;
        }
      `}</style>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className={isSubmitting ? 'loading-state' : ''}
      >
        <div className="url-form-container">
          <div className="form-header">
            <h3 className="form-title">
              <i className="bi bi-link-45deg me-2"></i>
              Product URL Analysis
            </h3>
            <p className="form-subtitle">
              Enter a product URL to extract details and generate video ads
            </p>
          </div>

          <Form onSubmit={handleSubmit}>
            <Row>
              <Col lg={8} className="mb-3">
                <InputGroup>
                  <InputGroup.Text>
                    <i className="bi bi-globe input-icon"></i>
                  </InputGroup.Text>
                  <Form.Control
                    type="url"
                    placeholder="https://example.com/product-page"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    disabled={isSubmitting}
                    className="url-input"
                    required
                  />
                </InputGroup>
              </Col>
              <Col lg={4} className="mb-3">
                <Button 
                  type="submit" 
                  disabled={isSubmitting || !url.trim()}
                  className="analyze-button w-100"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner size="sm" className="me-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <i className="bi bi-search me-2"></i>
                      Analyze Product
                    </>
                  )}
                </Button>
              </Col>
            </Row>
          </Form>

          {!isSubmitting && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              transition={{ duration: 0.3, delay: 0.2 }}
              className="examples-section"
            >
              <div className="examples-title">
                <i className="bi bi-lightbulb me-2"></i>
                Try these example URLs:
              </div>
              <div>
                {exampleUrls.map((example, index) => (
                  <span
                    key={index}
                    className="example-chip"
                    onClick={() => setUrl(example.url)}
                  >
                    {example.label}
                  </span>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </>
  );
}

